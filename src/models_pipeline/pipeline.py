import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import shap
import os
import matplotlib
import argparse

matplotlib.use('Agg') 

from sklearn.metrics import classification_report, roc_curve, auc, brier_score_loss, log_loss
from sklearn.calibration import calibration_curve
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV, GroupKFold, KFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

import tensorflow as tf
from tensorflow.keras import layers, models, backend, callbacks, regularizers
import keras_tuner as kt
import sys

from sklearn.metrics import precision_score, recall_score, f1_score 
import tensorflow.keras.backend as K

def log_progress(msg):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()
    with open(os.path.join(output_dir, "progress.log"), "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def log_and_write_report(msg):
    log_progress(msg)
    with open(report_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def save_and_print_report(y_true, y_pred, model_name):
    report_str = classification_report(y_true, y_pred)
    print(f"\n>>> {model_name} REPORT <<<\n{report_str}", flush=True)
    
    with open(report_file, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f">>> {model_name} REPORT <<<\n")
        f.write(f"{'='*60}\n")
        f.write(report_str + "\n")


# ==========================================
# 0. SETUP
# ==========================================

parser = argparse.ArgumentParser(description="Script for classifying Refactoring techniques.")
parser.add_argument(
    '--target', '-t', 
    type=str, 
    default="Change Parameter Type",
    help='Name of the refactoring target (e.g. "Extract Method"). Default: "Change Parameter Type"'
)
parser.add_argument(
    '--tune', 
    action='store_true', 
    help='If present, tuners will run (compute intensive)'
)

args = parser.parse_args()

target_col = args.target
RUN_NN_TUNING = args.tune


# Resolve DesigniteJava root dynamically (3 parents up from project_thesis/src/models_pipeline/pipeline.py to get DesigniteJava/)
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

base_report_dir = os.path.join(BASE_PATH, "classification_report")
output_dir = os.path.join(base_report_dir, target_col.replace(" ", "_"))
os.makedirs(output_dir, exist_ok=True)
report_file = os.path.join(output_dir, "all_classification_reports.txt")

log_progress(f"\n[INFO] Target: {target_col}")
log_progress(f"[INFO] Save Dir: {output_dir}\n")
np.random.seed(42)
tf.random.set_seed(42)


open(os.path.join(output_dir, "all_classification_reports.txt"), "w").close()


# ==========================================
# 1. LOADING & CLEANING
# ==========================================
log_progress("[PROGRESS] Loading data...")
df1 = pd.read_csv(os.path.join(BASE_PATH, "embedded", "dataset", "dataset.csv"))
df2 = pd.read_csv(os.path.join(BASE_PATH, "embedded", "dataset", "dataset_zeros.csv"))
df = pd.concat([df1, df2], axis=0).reset_index(drop=True)
#df = pd.read_csv(os.path.join(BASE_PATH, "ck_graph2vec_merged", "ck_graph2vec_merged_all.csv"))

df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.drop_duplicates().dropna().reset_index(drop=True)

# --- Distribution of all refactoring targets ---

log_progress("[PROGRESS] Generating overall targets distribution plot...")
original_targets = ["Change Variable Type", "Change Parameter Type", "Change Return Type",
                    "Extract Method", "Move Method", "Rename Method", "Rename Variable",
                    "Rename Parameter", "Extract Variable", "Add Parameter"]

plt.figure(figsize=(12, 6))
existing_targets = [t for t in original_targets if t in df.columns]
targets_counts = df[existing_targets].sum().sort_values(ascending=False)

sns.barplot(x=targets_counts.values, y=targets_counts.index, palette="viridis")
plt.title("Distribution of Refactoring Techniques (Full Cleaned Dataset)")
plt.xlabel("Occurrences")
plt.ylabel("Refactoring Technique")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "all_targets_distribution.png"))
plt.close()

# ==========================================
# 2. PRE-SPLIT FEATURE ENGINEERING
# ==========================================

meta_cols = ['Method', 'Project', 'Package', 'Class', 'Description', 'File', 'line', 'Line no']

if 'Smell' in df.columns:
    df['Smell_Density'] = df['Smell'].astype(str).apply(lambda x: 0 if x.lower() == 'none' else len(x.split(',')))

    groups = df['Project']
initial_drop = [col for col in meta_cols + original_targets if col in df.columns and col not in ['Project', 'Smell']]
X_temp = df.drop(columns=initial_drop)
y_all = df[target_col]

# ==========================================
# 3. PROJECT DIAGNOSTICS & RANDOM SPLIT
# ==========================================
log_and_write_report("[PROGRESS] Data distribution by project...")

stats = X_temp.copy()
stats['target'] = y_all
project_stats = stats.groupby(groups)['target'].agg(
    positives='sum', 
    total='count'
).reset_index()

project_stats['negatives'] = project_stats['total'] - project_stats['positives']
project_stats = project_stats.sort_values(by='positives', ascending=False)

log_and_write_report("\n" + project_stats.to_string(index=False) + "\n")

log_and_write_report("\n[PROGRESS] Standard Random Split (80/20)...")

X_tr_raw, X_te_raw, y_tr_raw, y_te_raw = train_test_split(
    X_temp, y_all, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_all
)

if 'Smell' in X_tr_raw.columns:
    smell_train_raw_series = X_tr_raw['Smell'].copy()
    original_test_smells = X_te_raw['Smell'].copy()

# --- Target distribution before balancing ---
log_progress("[PROGRESS] Generating target distribution plot (BEFORE balancing)...")
plt.figure(figsize=(6, 4))
sns.countplot(x=y_tr_raw, palette="pastel")
plt.title(f"Target Distribution Before Balancing\n({target_col})")
plt.xlabel("Class (0 = Negative, 1 = Positive)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "target_before_balancing.png"))
plt.close()

# ==========================================
# 4. TARGET ENCODING (SMELL RISK SCORE\)
# ==========================================
if 'Smell' in X_tr_raw.columns:
    log_progress("[PROGRESS] Calculating Smell Risk Score (Target Encoding con Smoothing)...")
    
    temp_tr = pd.DataFrame({'Smell': X_tr_raw['Smell'], 'target': y_tr_raw})
    global_mean = y_tr_raw.mean()
    counts = temp_tr.groupby('Smell')['target'].count()
    means = temp_tr.groupby('Smell')['target'].mean()
    smoothing_weight = 10 
    smoothed_means = (counts * means + smoothing_weight * global_mean) / (counts + smoothing_weight)
    
    X_tr_raw['Smell_Risk_Score'] = X_tr_raw['Smell'].map(smoothed_means).fillna(global_mean)
    X_te_raw['Smell_Risk_Score'] = X_te_raw['Smell'].map(smoothed_means).fillna(global_mean)
    
    X_tr_raw = X_tr_raw.drop(columns=['Smell'])
    X_te_raw = X_te_raw.drop(columns=['Smell'])

if 'Project' in X_tr_raw.columns:
    X_tr_raw = X_tr_raw.drop(columns=['Project'])
    X_te_raw = X_te_raw.drop(columns=['Project'])

X_tr_final = X_tr_raw.select_dtypes(include=[np.number, bool]).astype(np.float32)
X_te_final = X_te_raw.select_dtypes(include=[np.number, bool]).astype(np.float32)

# --- Dataset Features Summary ---
log_and_write_report("\n[PROGRESS] --- Dataset Features Summary (Post-Metadata & Encoding) ---")
all_cols = X_tr_final.columns.tolist()
emb_cols_summary = [c for c in all_cols if "emb_" in c]
classic_cols_summary = [c for c in all_cols if "emb_" not in c]

log_and_write_report(f"Total Features Active: {len(all_cols)}")
log_and_write_report(f"Classic Features ({len(classic_cols_summary)}):")

for col in classic_cols_summary:
    log_and_write_report(f"  - {col}")

if len(emb_cols_summary) > 0:
    log_and_write_report(f"Embeddings Features: {len(emb_cols_summary)} columns (Represented as a single embedding block)")
else:
    log_and_write_report("Embeddings Features: None detected.")
    
log_and_write_report("-" * 75 + "\n")

# --- Smell Risk Score per Smell Category ---
log_progress("[PROGRESS] Generating Smell Risk Score per Smell Category plot...")
plt.figure(figsize=(12, 6))

sorted_smells = smoothed_means.sort_values(ascending=False)

top_smells = sorted_smells.head(30)

sns.barplot(x=top_smells.index, y=top_smells.values, palette="magma")
plt.title(f"Smell Risk Score by Smell Combination\nTarget: {target_col}")
plt.xlabel("Smell Combination")
plt.ylabel("Computed Risk Score")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "smell_risk_score_by_smell.png"))
plt.close()


# ==========================================
# 5. MASTER SCALING & BALANCING (ACADEMIC SETTING)
# ==========================================
log_progress("[PROGRESS] Applying Master Scaling & Academic Balancing...")

master_scaler = StandardScaler()
master_scaler.fit(X_tr_final) 

X_tr_all_s = pd.DataFrame(master_scaler.transform(X_tr_final), index=X_tr_final.index, columns=X_tr_final.columns)
X_te_all_s = pd.DataFrame(master_scaler.transform(X_te_final), index=X_te_final.index, columns=X_te_final.columns)

def smart_balance(X_p, y_p): 
    pos_idx = y_p[y_p == 1].index
    neg_idx = y_p[y_p == 0].index
    n = min(len(pos_idx), len(neg_idx))
    
    c_pos = np.random.choice(pos_idx, n, replace=False)
    c_neg = np.random.choice(neg_idx, n, replace=False)
    idx = np.concatenate([c_pos, c_neg])
    np.random.shuffle(idx)
    
    return X_p.loc[idx], y_p.loc[idx]

X_train_s, y_train = smart_balance(X_tr_all_s, y_tr_raw)
X_test_s, y_test = smart_balance(X_te_all_s, y_te_raw)

smell_train_balanced = smell_train_raw_series.loc[X_train_s.index]

log_and_write_report(f"Balanced Train Set: {len(y_train)} | Balanced Test Set: {len(y_test)}")

# --- Box plots before and after scaling ---
log_progress("[PROGRESS] Generating box plots for scaling comparison...")
cols_to_plot = X_tr_final.columns[:10] 

fig, axes = plt.subplots(2, 1, figsize=(14, 12))

# Before Scaling
sns.boxplot(data=X_tr_final[cols_to_plot], ax=axes[0], orient="h", palette="Set2")
axes[0].set_title("Feature Distributions BEFORE Scaling (Sample of 10 Features)")
axes[0].set_xlabel("Original Values")

# After Scaling
sns.boxplot(data=X_tr_all_s[cols_to_plot], ax=axes[1], orient="h", palette="Set2")
axes[1].set_title("Feature Distributions AFTER Scaling (Sample of 10 Features)")
axes[1].set_xlabel("Scaled Values (Standardized)")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "boxplots_scaling_comparison.png"))
plt.close()

# --- Target distribution after balancing ---
log_progress("[PROGRESS] Generating target distribution plot (AFTER balancing)...")
plt.figure(figsize=(6, 4))
sns.countplot(x=y_train, palette="pastel")
plt.title(f"Target Distribution After Balancing\n({target_col})")
plt.xlabel("Class (0 = Negative, 1 = Positive)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "target_after_balancing.png"))
plt.close()

# ==========================================
# 6. HYPERPARAMETER TUNING (XGBoost)
# ==========================================
log_and_write_report("\n--- Searching for best params (XGBoost) ---")
param_dist = {
    'max_depth': [3, 4, 6],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.7, 0.9],
    'colsample_bytree': [0.3, 0.5]
}
xgb_search = XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42)
random_search = RandomizedSearchCV(xgb_search, param_distributions=param_dist, n_iter=5, 
                                   cv=3, scoring='f1', n_jobs=-1, random_state=42)

random_search.fit(X_train_s, y_train)
best_params = random_search.best_params_
log_and_write_report(f"Best params: {best_params}")

# ==========================================
# 7. FINAL TRAINING AND NEURAL NETWORKS
# ==========================================

class CorrectF1Metric(tf.keras.metrics.Metric):
    def __init__(self, name='f1_metric', **kwargs):
        super(CorrectF1Metric, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(tf.round(y_pred), tf.float32)
        self.tp.assign_add(tf.reduce_sum(y_true * y_pred))
        self.fp.assign_add(tf.reduce_sum((1 - y_true) * y_pred))
        self.fn.assign_add(tf.reduce_sum(y_true * (1 - y_pred)))

    def result(self):
        precision = self.tp / (self.tp + self.fp + K.epsilon())
        recall = self.tp / (self.tp + self.fn + K.epsilon())
        return 2 * precision * recall / (precision + recall + K.epsilon())

    def reset_state(self):
        for var in self.variables:
            var.assign(0.0)

models_map = {
    "XGBOOST": XGBClassifier(n_estimators=100, **best_params, random_state=42),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42),
    "LOG_REG": LogisticRegression(max_iter=1000, random_state=42)
}

results_proba = {}

# --- 7.1 Standard Models ---
for name, model in models_map.items():
    log_and_write_report(f"[PROGRESS] Training {name}...")
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    results_proba[name] = model.predict_proba(X_test_s)[:, 1]
    save_and_print_report(y_test, preds, name)

class UnbufferedProgress(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0: 
            log_and_write_report(f" -> [TRAINING] Epoch {epoch+1:03d} | Loss: {logs.get('loss'):.4f} | Val Loss: {logs.get('val_loss'):.4f}")

progress_tracker = UnbufferedProgress()
stop_early_strict = callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

# --- 7.2 Baseline Neural Network (Hardcoded) ---
log_and_write_report("\n--- Training Baseline Neural Network (Hardcoded) ---")
nn_baseline = models.Sequential([
    layers.Input(shape=(X_train_s.shape[1],)),
    layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.4), 
    layers.Dense(8, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

nn_baseline.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', CorrectF1Metric()])

history_baseline = nn_baseline.fit(
    X_train_s, y_train, 
    epochs=100, 
    batch_size=16, 
    validation_split=0.15,
    callbacks=[stop_early_strict, progress_tracker],
    verbose=0 
)

base_probs = nn_baseline.predict(X_test_s, verbose=0).ravel()
base_preds = (base_probs > 0.5).astype(int)
results_proba["NN_BASELINE"] = base_probs

save_and_print_report(y_test, base_preds, "NN_BASELINE")


# WEIGHT AUTOPSY
log_and_write_report("\n[DEBUG] --- Weight Autopsy of the Full Network ---")

input_weights = nn_baseline.layers[0].get_weights()[0]
feature_strength = np.sum(np.abs(input_weights), axis=1)

emb_cols = [c for c in X_train_s.columns if "emb_" in c]
classic_cols = [c for c in X_train_s.columns if "emb_" not in c]

idx_emb = [X_train_s.columns.get_loc(c) for c in emb_cols]
idx_classic = [X_train_s.columns.get_loc(c) for c in classic_cols]

total_strength_classic = np.sum(feature_strength[idx_classic])
total_strength_emb = np.sum(feature_strength[idx_emb])

avg_strength_single_classic = np.mean(feature_strength[idx_classic])
avg_strength_single_emb = np.mean(feature_strength[idx_emb])

log_and_write_report(f" -> TOTAL Strength absorbed by {len(classic_cols)} Classic features: {total_strength_classic:.4f}")
log_and_write_report(f" -> TOTAL Strength absorbed by {len(emb_cols)} Embeddings: {total_strength_emb:.4f}")

# --- 7.3 Targeted Tuning Neural Network ---
log_progress("\n--- Starting Keras Tuner (Optimizing for F1-Score) ---")

def build_targeted_model(hp):
    model = models.Sequential()
    model.add(layers.Input(shape=(X_train_s.shape[1],)))
    
    hp_units_1 = hp.Int('units_l1', min_value=8, max_value=32, step=8) 
    hp_l2_rate = hp.Float('l2_reg', min_value=1e-3, max_value=5e-2, sampling='log')
    model.add(layers.Dense(units=hp_units_1, activation='relu', kernel_regularizer=regularizers.l2(hp_l2_rate)))
    
    model.add(layers.Dropout(hp.Float('dropout_l1', min_value=0.2, max_value=0.5, step=0.1)))
    
    hp_units_2 = hp.Int('units_l2', min_value=4, max_value=16, step=4)
    model.add(layers.Dense(units=hp_units_2, activation='relu'))
    model.add(layers.Dropout(hp.Float('dropout_l2', min_value=0.1, max_value=0.3, step=0.1)))
    
    model.add(layers.Dense(1, activation='sigmoid'))
    
    lr = hp.Float('learning_rate', min_value=1e-4, max_value=5e-3, sampling='log')
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr), 
        loss='binary_crossentropy', 
        metrics=['accuracy', CorrectF1Metric()]
    )
    return model

tuner = kt.BayesianOptimization(
    build_targeted_model,
    objective=kt.Objective("val_f1_metric", direction="max"),
    max_trials=20, 
    directory=os.path.join(output_dir, 'tuning_dir_targeted'), 
    project_name='nn_f1_opt', 
    overwrite=RUN_NN_TUNING
)

if RUN_NN_TUNING:
    early_stop_tuner = callbacks.EarlyStopping(
        monitor='val_f1_metric', 
        mode='max', 
        patience=15, 
        restore_best_weights=True
    )
    
    tuner.search(
        X_train_s, y_train, 
        epochs=80, 
        batch_size=16, 
        validation_split=0.15, 
        callbacks=[early_stop_tuner], 
        verbose=0
    )

best_hps_targeted = {
    'units_l1': 16,
    'units_l2': 8,
    'dropout_l1': 0.4,
    'dropout_l2': 0.2,
    'learning_rate': 1e-3,
    'l2_reg': 0.01
}

try:
    best_hps_targeted = tuner.get_best_hyperparameters(num_trials=1)[0]
    log_progress(f"[INFO] Best Tuned HPs: L1={best_hps_targeted.get('units_l1')}, L2={best_hps_targeted.get('units_l2')}, LR={best_hps_targeted.get('learning_rate'):.4f}")
    
    nn_tuned_targeted = tuner.hypermodel.build(best_hps_targeted)
    
    early_stop_final = callbacks.EarlyStopping(
        monitor='val_f1_metric', mode='max', patience=15, restore_best_weights=True
    )
    
    nn_tuned_targeted.fit(
        X_train_s, y_train, 
        epochs=100, 
        batch_size=16, 
        validation_split=0.15, 
        callbacks=[early_stop_final, progress_tracker], 
        verbose=0
    )
    
    tuned_probs = nn_tuned_targeted.predict(X_test_s, verbose=0).ravel()
    results_proba["NN_TUNED_TARGETED"] = tuned_probs
    
    save_and_print_report(y_test, (tuned_probs > 0.5).astype(int), "NN_TUNED_TARGETED")

except Exception as e:
    log_progress(f"[ERROR] Tuner issue: {e}")

# ==========================================
# 8. ABLATION STUDY
# ==========================================
log_and_write_report("\n--- Starting Ablation Study (No Leakage) ---")

X_tr_abl_s = X_train_s[classic_cols]
X_te_abl_s = X_test_s[classic_cols]

xgb_abl = XGBClassifier(n_estimators=100, **best_params, random_state=42)
xgb_abl.fit(X_tr_abl_s, y_train)
results_proba["XGB_ABLATED"] = xgb_abl.predict_proba(X_te_abl_s)[:, 1]

xgb_abl_preds = (results_proba["XGB_ABLATED"] > 0.5).astype(int)
save_and_print_report(y_test, xgb_abl_preds, "XGB_ABLATED")

nn_abl = models.Sequential([
    layers.Input(shape=(X_tr_abl_s.shape[1],)),
    layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.4), layers.Dense(8, activation='relu'), layers.Dropout(0.2), layers.Dense(1, activation='sigmoid')
])
nn_abl.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', CorrectF1Metric()])

history_ablated = nn_abl.fit(X_tr_abl_s, y_train, epochs=100, batch_size=16, validation_split=0.15, callbacks=[stop_early_strict, progress_tracker], verbose=0)
results_proba["NN_ABLATED"] = nn_abl.predict(X_te_abl_s, verbose=0).ravel()

save_and_print_report(y_test, (results_proba["NN_ABLATED"] > 0.5).astype(int), "NN_ABLATED (HARDCODED)")

# --- 8.2 Tuning Neural Network Ablated (CONDITIONAL) ---
log_progress("\n--- Setting up Keras Tuner for ABLATED NN ---")

def build_ablated_model(hp):
    model = models.Sequential()
    model.add(layers.Input(shape=(X_tr_abl_s.shape[1],)))
    
    hp_units_1 = hp.Int('units_l1', min_value=8, max_value=32, step=8)
    hp_l2_rate = hp.Float('l2_reg', min_value=1e-3, max_value=5e-2, sampling='log')
    model.add(layers.Dense(units=hp_units_1, activation='relu', kernel_regularizer=regularizers.l2(hp_l2_rate)))
    model.add(layers.Dropout(hp.Float('dropout_l1', min_value=0.2, max_value=0.5, step=0.1)))
    
    hp_units_2 = hp.Int('units_l2', min_value=4, max_value=16, step=4)
    model.add(layers.Dense(units=hp_units_2, activation='relu'))
    model.add(layers.Dropout(hp.Float('dropout_l2', min_value=0.1, max_value=0.3, step=0.1)))
    
    model.add(layers.Dense(1, activation='sigmoid'))
    
    lr = hp.Float('learning_rate', min_value=1e-4, max_value=5e-3, sampling='log')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy', CorrectF1Metric()])
    return model

tuner_ablated = kt.BayesianOptimization(
    build_ablated_model,
    objective=kt.Objective("val_f1_metric", direction="max"),
    max_trials=20, 
    directory=os.path.join(output_dir, 'tuning_dir_ablated'),
    project_name='nn_ablated_f1_opt',
    overwrite=RUN_NN_TUNING
)

if RUN_NN_TUNING:
    log_progress(" -> [TUNING] Executing search for best Ablated NN parameters...")
    early_stop_tuner_abl = callbacks.EarlyStopping(monitor='val_f1_metric', mode='max', patience=15, restore_best_weights=True)
    tuner_ablated.search(X_tr_abl_s, y_train, epochs=80, batch_size=16, validation_split=0.15, callbacks=[early_stop_tuner_abl], verbose=0)

try:
    best_hps_ablated = tuner_ablated.get_best_hyperparameters(num_trials=1)[0]
    log_progress(f"[INFO] Best Tuned HPs (ABLATED): L1={best_hps_ablated.get('units_l1')}, L2={best_hps_ablated.get('units_l2')}, LR={best_hps_ablated.get('learning_rate'):.4f}")

    nn_tuned_ablated = tuner_ablated.hypermodel.build(best_hps_ablated)
    early_stop_final_abl = callbacks.EarlyStopping(monitor='val_f1_metric', mode='max', patience=15, restore_best_weights=True)

    nn_tuned_ablated.fit(X_tr_abl_s, y_train, epochs=100, batch_size=16, validation_split=0.15, callbacks=[early_stop_final_abl, progress_tracker], verbose=0)

    tuned_abl_probs = nn_tuned_ablated.predict(X_te_abl_s, verbose=0).ravel()
    results_proba["NN_TUNED_ABLATED"] = tuned_abl_probs
    
    save_and_print_report(y_test, (tuned_abl_probs > 0.5).astype(int), "NN_TUNED_ABLATED")

except Exception as e:
    log_progress(f"[INFO] Tuned Ablated NN skipped or not available (no previous tuning found).")

# --- 8.3 ABLATION: EMBEDDINGS ONLY ---
log_and_write_report("\n--- Starting Ablation Study (Embeddings Only) ---")
X_tr_emb_s = X_train_s[emb_cols]
X_te_emb_s = X_test_s[emb_cols]

# --- 8.3.1 Neural Network Embeddings Only (HARDCODED) ---
log_progress("[PROGRESS] Training EMBEDDINGS ONLY Neural Network (Hardcoded)...")
nn_emb_only = models.Sequential([
    layers.Input(shape=(X_tr_emb_s.shape[1],)),
    layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.4), layers.Dense(8, activation='relu'), layers.Dropout(0.2), layers.Dense(1, activation='sigmoid')
])
nn_emb_only.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', CorrectF1Metric()])

history_emb_only = nn_emb_only.fit(X_tr_emb_s, y_train, epochs=100, batch_size=16, validation_split=0.15, callbacks=[stop_early_strict, progress_tracker], verbose=0)
results_proba["NN_EMBEDDINGS_ONLY"] = nn_emb_only.predict(X_te_emb_s, verbose=0).ravel()

save_and_print_report(y_test, (results_proba["NN_EMBEDDINGS_ONLY"] > 0.5).astype(int), "NN_EMBEDDINGS_ONLY (HARDCODED)")

# --- 8.3.2 Tuning Neural Network Embeddings Only  ---
log_progress("\n--- Setting up Keras Tuner for EMBEDDINGS ONLY NN ---")

def build_emb_model(hp):
    model = models.Sequential()
    model.add(layers.Input(shape=(X_tr_emb_s.shape[1],)))
    
    hp_units_1 = hp.Int('units_l1', min_value=8, max_value=32, step=8)
    hp_l2_rate = hp.Float('l2_reg', min_value=1e-3, max_value=5e-2, sampling='log')
    model.add(layers.Dense(units=hp_units_1, activation='relu', kernel_regularizer=regularizers.l2(hp_l2_rate)))
    model.add(layers.Dropout(hp.Float('dropout_l1', min_value=0.2, max_value=0.5, step=0.1)))
    
    hp_units_2 = hp.Int('units_l2', min_value=4, max_value=16, step=4)
    model.add(layers.Dense(units=hp_units_2, activation='relu'))
    model.add(layers.Dropout(hp.Float('dropout_l2', min_value=0.1, max_value=0.3, step=0.1)))
    
    model.add(layers.Dense(1, activation='sigmoid'))
    
    lr = hp.Float('learning_rate', min_value=1e-4, max_value=5e-3, sampling='log')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy', CorrectF1Metric()])
    return model

tuner_emb = kt.BayesianOptimization(
    build_emb_model,
    objective=kt.Objective("val_f1_metric", direction="max"),
    max_trials=20, 
    directory=os.path.join(output_dir, 'tuning_dir_emb'),
    project_name='nn_emb_f1_opt',
    overwrite=RUN_NN_TUNING
)

if RUN_NN_TUNING:
    log_progress(" -> [TUNING] Executing search for best Embeddings Only NN parameters...")
    early_stop_tuner_emb = callbacks.EarlyStopping(monitor='val_f1_metric', mode='max', patience=15, restore_best_weights=True)
    tuner_emb.search(X_tr_emb_s, y_train, epochs=80, batch_size=16, validation_split=0.15, callbacks=[early_stop_tuner_emb], verbose=0)

try:
    best_hps_emb = tuner_emb.get_best_hyperparameters(num_trials=1)[0]
    log_progress(f"[INFO] Best Tuned HPs (EMBEDDINGS ONLY): L1={best_hps_emb.get('units_l1')}, L2={best_hps_emb.get('units_l2')}, LR={best_hps_emb.get('learning_rate'):.4f}")

    nn_tuned_emb = tuner_emb.hypermodel.build(best_hps_emb)
    early_stop_final_emb = callbacks.EarlyStopping(monitor='val_f1_metric', mode='max', patience=15, restore_best_weights=True)

    nn_tuned_emb.fit(X_tr_emb_s, y_train, epochs=100, batch_size=16, validation_split=0.15, callbacks=[early_stop_final_emb, progress_tracker], verbose=0)

    tuned_emb_probs = nn_tuned_emb.predict(X_te_emb_s, verbose=0).ravel()
    results_proba["NN_TUNED_EMBEDDINGS"] = tuned_emb_probs
    
    save_and_print_report(y_test, (tuned_emb_probs > 0.5).astype(int), "NN_TUNED_EMBEDDINGS")

except Exception as e:
    log_progress(f"[INFO] Tuned Embeddings Only NN skipped or not available (no previous tuning found).")

# ==========================================
# 8.5 STACKING ENSEMBLE (LATE FUSION)
# ==========================================
log_and_write_report("\n--- Starting Stacking Ensemble (Late Fusion) ---")

X_train_classiche = X_train_s[classic_cols].values
X_train_embeddings = X_train_s[emb_cols].values
X_test_classiche = X_test_s[classic_cols].values
X_test_embeddings = X_test_s[emb_cols].values

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds_xgb = np.zeros(len(y_train))
oof_preds_nn = np.zeros(len(y_train))

if 'Smell_Risk_Score' in classic_cols:
    smell_idx_in_classic = classic_cols.index('Smell_Risk_Score')
else:
    smell_idx_in_classic = None

log_and_write_report("[PROGRESS] Generating OOF predictions via 5-Fold CV...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_classiche)):
    log_progress(f" -> Processing Fold {fold+1}/5...")
    
    X_tr_c = X_train_classiche[train_idx].copy()
    X_val_c = X_train_classiche[val_idx].copy()
    X_tr_e = X_train_embeddings[train_idx]
    X_val_e = X_train_embeddings[val_idx]
    y_tr, y_val = y_train.values[train_idx], y_train.values[val_idx]
    
    if smell_idx_in_classic is not None:
        smells_train_fold = smell_train_balanced.values[train_idx]
        smells_val_fold = smell_train_balanced.values[val_idx]
        
        temp_fold = pd.DataFrame({'Smell': smells_train_fold, 'target': y_tr})
        fold_mean = y_tr.mean()
        counts_fold = temp_fold.groupby('Smell')['target'].count()
        means_fold = temp_fold.groupby('Smell')['target'].mean()
        
        smoothed_means_fold = (counts_fold * means_fold + 10 * fold_mean) / (counts_fold + 10)
        
        raw_tr_smell = pd.Series(smells_train_fold).map(smoothed_means_fold).fillna(fold_mean).values.reshape(-1, 1)
        raw_val_smell = pd.Series(smells_val_fold).map(smoothed_means_fold).fillna(fold_mean).values.reshape(-1, 1)
        
        from sklearn.preprocessing import StandardScaler
        fold_scaler = StandardScaler()
        X_tr_c[:, smell_idx_in_classic] = fold_scaler.fit_transform(raw_tr_smell).ravel()
        X_val_c[:, smell_idx_in_classic] = fold_scaler.transform(raw_val_smell).ravel()

    m_xgb = XGBClassifier(n_estimators=100, **best_params, random_state=42)
    m_xgb.fit(X_tr_c, y_tr)
    oof_preds_xgb[val_idx] = m_xgb.predict_proba(X_val_c)[:, 1]
    
    m_nn = models.Sequential([
        layers.Input(shape=(X_tr_e.shape[1],)),
        layers.Dense(16, activation='relu'), layers.Dropout(0.2), layers.Dense(8, activation='relu'), layers.Dense(1, activation='sigmoid')
    ])
    m_nn.compile(optimizer='adam', loss='binary_crossentropy')
    m_nn.fit(X_tr_e, y_tr, epochs=50, batch_size=16, verbose=0, validation_data=(X_val_e, y_val), callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])
    
    oof_preds_nn[val_idx] = m_nn.predict(X_val_e, verbose=0).ravel()
    tf.keras.backend.clear_session()

# --- Meta-Learner ---
X_meta_train = np.column_stack((oof_preds_xgb, oof_preds_nn))
meta_learner = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
meta_learner.fit(X_meta_train, y_train)

importances = meta_learner.feature_importances_
log_and_write_report(f"[INFO] Meta-Learner Importances -> XGB (Classic): {importances[0]:.4f}, NN (Embeddings): {importances[1]:.4f}")

# Final inference
xgb_expert_final = XGBClassifier(n_estimators=100, **best_params, random_state=42)
xgb_expert_final.fit(X_train_classiche, y_train)

nn_expert_final = models.Sequential([
    layers.Input(shape=(X_train_embeddings.shape[1],)),
    layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.4), layers.Dense(8, activation='relu'), layers.Dense(1, activation='sigmoid')
])
nn_expert_final.compile(optimizer='adam', loss='binary_crossentropy')
nn_expert_final.fit(X_train_embeddings, y_train, epochs=50, batch_size=16, verbose=0)

test_preds_xgb = xgb_expert_final.predict_proba(X_test_classiche)[:, 1]
test_preds_nn = nn_expert_final.predict(X_test_embeddings, verbose=0).ravel() 

X_meta_test = np.column_stack((test_preds_xgb, test_preds_nn))
stacking_probs = meta_learner.predict_proba(X_meta_test)[:, 1]
results_proba["STACKING_ENSEMBLE"] = stacking_probs

save_and_print_report(y_test, (stacking_probs > 0.5).astype(int), "STACKING_ENSEMBLE")

# ==========================================
# 8.7 PCA DIMENSIONALITY REDUCTION
# ==========================================
log_and_write_report("\n--- Starting PCA Reduction + XGBoost ---")
n_pca = 3
pca = PCA(n_components=n_pca, random_state=42)
train_pca = pca.fit_transform(X_train_embeddings)
test_pca = pca.transform(X_test_embeddings)

X_tr_pca_combined = np.hstack((X_train_classiche, train_pca))
X_te_pca_combined = np.hstack((X_test_classiche, test_pca))

xgb_pca = XGBClassifier(n_estimators=100, **best_params, random_state=42)
xgb_pca.fit(X_tr_pca_combined, y_train)
results_proba["XGB_WITH_PCA_EMB"] = xgb_pca.predict_proba(X_te_pca_combined)[:, 1]
save_and_print_report(y_test, (results_proba["XGB_WITH_PCA_EMB"] > 0.5).astype(int), "XGB_WITH_PCA_EMB")

# ==========================================
# 8.8 SELF-SUPERVISE LEARNING
# ==========================================
log_and_write_report("\n" + "="*60 + "\n SELF-SUPERVISED LEARNING (SSL)\n" + "="*60)

# --- DATA PREPARATION ---
indices_ft = y_train.index
X_ssl_raw_pool = X_tr_final.drop(index=indices_ft, errors='ignore')


ssl_scaler = StandardScaler()
X_ssl_pool_scaled = pd.DataFrame(
    ssl_scaler.fit_transform(X_ssl_raw_pool), 
    columns=X_ssl_raw_pool.columns
)


input_dim = X_ssl_pool_scaled.shape[1]
encoding_dim = 64

# --- PRE-TASK ARCHTECTURES ---

# A. Autoencoder
def build_autoencoder(dim, latent_dim):
    input_layer = layers.Input(shape=(dim,))
    # Larger layers to handle the embeddings
    enc = layers.Dense(64, activation='relu')(input_layer) 
    bottleneck = layers.Dense(latent_dim, activation='relu', name='encoder_output')(enc)
    dec = layers.Dense(64, activation='relu')(bottleneck)
    out = layers.Dense(dim, activation='linear')(dec)
    
    ae = models.Model(input_layer, out)
    encoder = models.Model(input_layer, bottleneck)
    ae.compile(optimizer='adam', loss='mse')
    return ae, encoder

# B. Contrastive Model
class ContrastiveModel(models.Model):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.noise_factor = 0.1

    def train_step(self, data):
        x = data[0] if isinstance(data, tuple) else data
        x1 = x + self.noise_factor * tf.random.normal(shape=tf.shape(x))
        x2 = x + self.noise_factor * tf.random.normal(shape=tf.shape(x))
        
        with tf.GradientTape() as tape:
            z1 = self.encoder(x1)
            z2 = self.encoder(x2)
            loss = tf.reduce_mean(tf.square(z1 - z2)) 
        
        gradients = tape.gradient(loss, self.encoder.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.encoder.trainable_variables))
        return {"contrastive_loss": loss}

# --- 3.  PRE-TASK EXECUTION---
log_progress("[SSL] Executing Pre-task: Training Autoencoder...")
ae_full, encoder_ae = build_autoencoder(input_dim, encoding_dim)
ae_full.fit(X_ssl_pool_scaled.values, X_ssl_pool_scaled.values, epochs=30, batch_size=64, validation_split=0.1, verbose=0)
log_progress("[SSL] Executing Pre-task: Training Contrastive Encoder...")
contrastive_base = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(encoding_dim, activation='linear')
])
c_model = ContrastiveModel(contrastive_base)
c_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
c_model.fit(X_ssl_pool_scaled.values, epochs=25, batch_size=64, verbose=0)


# --- 4. FINE-TUNING ---

def fine_tune_with_warmup(base_encoder, name, hps):
    
    
    units_l1 = hps.get('units_l1')
    units_l2 = hps.get('units_l2')
    drop_l1 = hps.get('dropout_l1')
    drop_l2 = hps.get('dropout_l2')
    
    base_encoder.trainable = False 
    model = models.Sequential([
        base_encoder, 
        
        layers.Dense(units_l1, activation='relu'),
        layers.Dropout(drop_l1),
        layers.Dense(units_l2, activation='relu'),
        layers.Dropout(drop_l2),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy', CorrectF1Metric()])
    
    log_progress(f" -> Phase 1: Warm-up")
    model.fit(X_train_s.values, y_train.values, epochs=10, batch_size=16, validation_split=0.15, verbose=0)
    
    base_encoder.trainable = True

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy', CorrectF1Metric()])
    
    log_progress(f" -> Phase 2: unfreezing wheights and Fine-Tuning end-to-end...")
    early_stop_ft = callbacks.EarlyStopping(monitor='val_f1_metric', mode='max', patience=12, restore_best_weights=True)
    
    model.fit(X_train_s.values, y_train.values, epochs=50, batch_size=16, validation_split=0.15, 
          callbacks=[early_stop_ft, progress_tracker], verbose=0)
    
    return model

ft_ae_nn = fine_tune_with_warmup(encoder_ae, "Autoencoder", best_hps_targeted)
ft_cont_nn = fine_tune_with_warmup(contrastive_base, "Contrastive", best_hps_targeted)

probs_ssl_ae = ft_ae_nn.predict(X_test_s, verbose=0).ravel()
probs_ssl_cont = ft_cont_nn.predict(X_test_s, verbose=0).ravel()

results_proba["SSL_AUTOENCODER_FT"] = probs_ssl_ae
results_proba["SSL_CONTRASTIVE_FT"] = probs_ssl_cont

save_and_print_report(y_test, (probs_ssl_ae > 0.5).astype(int), "SSL_AUTOENCODER_FT")
save_and_print_report(y_test, (probs_ssl_cont > 0.5).astype(int), "SSL_CONTRASTIVE_FT")


# ==========================================
# 9. PLOTS GENERATION
# ==========================================
log_progress("\n--- Generating Plots ---")

# 9.0 Feature Importance XGBoost Standard
log_progress("[PROGRESS] Computing Feature Importance for Standard XGBoost...")
xgb_standard = models_map["XGBOOST"]
if hasattr(xgb_standard, 'feature_importances_'):
    importances_xgb = xgb_standard.feature_importances_
    # Select the top 20 features to avoid unreadable plots
    top_n = min(20, len(importances_xgb))
    indices_xgb = np.argsort(importances_xgb)[::-1][:top_n]
    
    plt.figure(figsize=(12,8))
    plt.title(f"XGBoost Feature Importance (Top {top_n})")
    plt.bar(range(top_n), importances_xgb[indices_xgb], align="center", color='teal')
    plt.xticks(range(top_n), [X_train_s.columns[i] for i in indices_xgb], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance_xgb.png"))
    plt.close()
plt.figure(figsize=(8,6))
allowed_roc_models = ["XGBOOST", "SVM", "LOG_REG", "NN_TUNED_TARGETED"]
for name, proba in results_proba.items():
    if name in allowed_roc_models:
        fpr, tpr, _ = roc_curve(y_test, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.2f})")
plt.plot([0,1],[0,1], 'k--')
plt.legend()
plt.savefig(os.path.join(output_dir, "roc_comparison.png"))
plt.close()

# Calibration Curves & Metrics
log_progress("[PROGRESS] Generating Calibration Curves & Computing Calibration Metrics...")

log_and_write_report("\n" + "="*60)
log_and_write_report(" CALIBRATION METRICS (Log Loss & Brier Score)")
log_and_write_report("="*60)
log_and_write_report(f"{'MODEL':<25} | {'LOG LOSS':<10} | {'BRIER SCORE':<12}")
log_and_write_report("-" * 55)

for name, proba in results_proba.items():
    loss_ll = log_loss(y_test, proba)
    loss_bs = brier_score_loss(y_test, proba)
    log_and_write_report(f"{name:<25} | {loss_ll:.4f}     | {loss_bs:.4f}")

log_and_write_report("-" * 55 + "\n")

# Plot 1: XGBoost Comparison (Standard vs Ablated)
log_progress("[PROGRESS] Generating Calibration Curve: XGBoost vs XGB_ABLATED...")
plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")

xgb_models = ["XGBOOST", "XGB_ABLATED"]
for name in xgb_models:
    if name in results_proba:
        proba = results_proba[name]
        loss_ll = log_loss(y_test, proba)
        loss_bs = brier_score_loss(y_test, proba)
        prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10)
        plt.plot(prob_pred, prob_true, "s-", label=f"{name} (Brier={loss_bs:.3f}, LL={loss_ll:.3f})")

plt.ylabel("Fraction of positives")
plt.xlabel("Mean predicted probability")
plt.ylim([-0.05, 1.05])
plt.legend(loc="lower right")
plt.title(f"XGBoost Calibration Comparison - Target: {target_col}")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "calibration_xgb_comparison.png"))
plt.close()

# Plot 2: Neural Network Comparison (Target vs Ablated)
log_progress("[PROGRESS] Generating Calibration Curve: NN Target vs NN Ablated...")
plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")

nn_target_name = "NN_TUNED_TARGETED" if "NN_TUNED_TARGETED" in results_proba else "NN_BASELINE"
nn_ablated_name = "NN_TUNED_ABLATED" if "NN_TUNED_ABLATED" in results_proba else "NN_ABLATED"

nn_models = [nn_target_name, nn_ablated_name]
for name in nn_models:
    if name in results_proba:
        proba = results_proba[name]
        loss_ll = log_loss(y_test, proba)
        loss_bs = brier_score_loss(y_test, proba)
        prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10)
        plt.plot(prob_pred, prob_true, "s-", label=f"{name} (Brier={loss_bs:.3f}, LL={loss_ll:.3f})")

plt.ylabel("Fraction of positives")
plt.xlabel("Mean predicted probability")
plt.ylim([-0.05, 1.05])
plt.legend(loc="lower right")
plt.title(f"Neural Network Calibration Comparison - Target: {target_col}")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "calibration_nn_comparison.png"))
plt.close()



# Learning Curves F1
plt.figure(figsize=(12, 7))
f1_train_base = history_baseline.history['f1_metric']
f1_val_base = history_baseline.history['val_f1_metric']
epochs_base = range(1, len(f1_train_base) + 1)

f1_train_abl = history_ablated.history['f1_metric']
f1_val_abl = history_ablated.history['val_f1_metric']
epochs_abl = range(1, len(f1_train_abl) + 1)

plt.plot(epochs_base, f1_train_base, label='Train F1 (With Embeddings)', color='royalblue', linestyle='--', alpha=0.6)
plt.plot(epochs_base, f1_val_base, label='Val F1 (With Embeddings)', color='blue', linewidth=2.5)
plt.plot(epochs_abl, f1_train_abl, label='Train F1 (No Embeddings)', color='lightcoral', linestyle='--', alpha=0.6)
plt.plot(epochs_abl, f1_val_abl, label='Val F1 (No Embeddings)', color='red', linewidth=2.5)

plt.title("Neural Network F1-Score: Embeddings vs Classic Metrics")
plt.legend(); plt.grid(True)
plt.savefig(os.path.join(output_dir, "nn_f1_comparison_curve.png"), bbox_inches='tight')
plt.close()

# UMAP
emb = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(X_test_s)
plt.figure(figsize=(10,8))
plt.scatter(emb[:, 0], emb[:, 1], c=y_test, cmap='coolwarm', s=10)
plt.savefig(os.path.join(output_dir, "umap_test.png"))
plt.close()

log_progress("[PROGRESS] Computing Feature Importance values for NN (via SHAP DeepExplainer)...")

try:
    background_sample = X_train_s.sample(n=100, random_state=42).values
    test_sample = X_test_s.sample(n=100, random_state=42).values
    
    explainer_nn = shap.DeepExplainer(nn_baseline, background_sample)
    feature_importance_nn_vals = explainer_nn.shap_values(test_sample)
    
    fi_vals_to_plot = feature_importance_nn_vals[0] if isinstance(feature_importance_nn_vals, list) else feature_importance_nn_vals
    
    if fi_vals_to_plot.ndim == 3:
        fi_vals_to_plot = np.squeeze(fi_vals_to_plot, axis=-1)
    
    plt.figure(figsize=(10,8))
    shap.summary_plot(fi_vals_to_plot, pd.DataFrame(test_sample, columns=X_train_s.columns), plot_type="bar", show=False)
    plt.title("Feature Importance (Neural Network via SHAP)")
    plt.savefig(os.path.join(output_dir, "feature_importance_nn_summary.png"), bbox_inches='tight')
    plt.close()
    
    fi_matrix = np.abs(fi_vals_to_plot).mean(axis=0)

    importanza_emb, importanza_classica = 0.0, 0.0
    for i, col_name in enumerate(X_train_s.columns):
        if "emb_" in str(col_name):
            importanza_emb += float(fi_matrix[i])
        else:
            importanza_classica += float(fi_matrix[i])

    plt.figure(figsize=(8,8))
    plt.pie([importanza_classica, importanza_emb], labels=['Classic Metrics', 'Embeddings'], autopct='%1.1f%%', startangle=90)
    plt.title("Macro Feature Importance for NN")
    plt.savefig(os.path.join(output_dir, "feature_importance_macro_pie.png"), bbox_inches='tight')
    plt.close()

except Exception as e:
    log_progress(f"[ERROR] Feature Importance (SHAP) calculation failed: {e}")

# ==========================================
# 10. SLICE ANALYSIS: PERFORMANCE PER SMELL
# ==========================================
log_and_write_report("\n[PROGRESS] --- Metrics Analysis per Single Smell Category ---")

aligned_smells = original_test_smells.loc[y_test.index]

df_analysis = pd.DataFrame({
    'Smell': aligned_smells.values, 
    'Actual_Target': y_test.values,
    'Prediction': (stacking_probs > 0.5).astype(int) 
})

log_and_write_report(f"{'SMELL':<35} | {'F1':<5} | {'PREC':<5} | {'REC':<5} | {'SUPPORT (Positives)'}")
log_and_write_report("-" * 75)

for smell_name, group in df_analysis.groupby('Smell'):
    y_true_g = group['Actual_Target']
    y_pred_g = group['Prediction']
    
    total_support = len(group)
    actual_positives = sum(y_true_g == 1)
    
    if actual_positives == 0:
        continue
        
    prec = precision_score(y_true_g, y_pred_g, zero_division=0)
    rec = recall_score(y_true_g, y_pred_g, zero_division=0)
    f1 = f1_score(y_true_g, y_pred_g, zero_division=0)
    
    log_and_write_report(f"{str(smell_name)[:34]:<35} | {f1:.2f}  | {prec:.2f}  | {rec:.2f}  | {total_support} ({actual_positives} true refactorings)")
    
log_and_write_report("-" * 75)  
log_progress(f"\n[DONE] Script done. Work saved in: {output_dir}")