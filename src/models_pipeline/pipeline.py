import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import shap
import os
import matplotlib

matplotlib.use('Agg') 

from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models, backend, callbacks, regularizers
import keras_tuner as kt

# ==========================================
# 0. SETUP
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
target_col = "Change Variable Type" 

print(f"\n[INFO] Target: {target_col}", flush=True)
print(f"[INFO] Save Dir: {script_dir}\n", flush=True)
np.random.seed(42)
tf.random.set_seed(42)

# ==========================================
# 1. LOADING & CLEANING
# ==========================================
print("[PROGRESS] Loading data...", flush=True)
df1 = pd.read_csv(r"D:\papersEvolution\DesigniteJava\embedded\dataset\dataset.csv")
df2 = pd.read_csv(r"D:\papersEvolution\DesigniteJava\embedded\dataset\dataset_zeros.csv")
df = pd.concat([df1, df2], axis=0).reset_index(drop=True)

df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.drop_duplicates().dropna().reset_index(drop=True)

# ==========================================
# 2. PRE-SPLIT FEATURE ENGINEERING
# ==========================================
target_originali = ["Change Variable Type", "Change Parameter Type", "Change Return Type",
                    "Extract Method", "Move Method", "Rename Method", "Rename Variable",
                    "Rename Parameter", "Extract Variable", "Add Parameter"]
meta_cols = ['Method', 'Project', 'Package', 'Class', 'Description', 'File', 'line', 'Line no']

if 'Smell' in df.columns:
    df['Smell_Density'] = df['Smell'].astype(str).apply(lambda x: 0 if x.lower() == 'none' else len(x.split(',')))

groups = df['Project']
initial_drop = [col for col in meta_cols + target_originali if col in df.columns and col not in ['Project', 'Smell']]
X_temp = df.drop(columns=initial_drop)
y_all = df[target_col]

# ==========================================
# 3. PROJECT DIAGNOSTICS & RANDOM SPLIT
# ==========================================
print("[PROGRESS] Data distribution by project...", flush=True)

stats = X_temp.copy()
stats['target'] = y_all
project_stats = stats.groupby(groups)['target'].agg(
    positives='sum', 
    total='count'
).reset_index()

project_stats['negatives'] = project_stats['total'] - project_stats['positives']
project_stats = project_stats.sort_values(by='positives', ascending=False)

print("\n[DEBUG] Top 10 projects by number of " + target_col + " refactoring technique:", flush=True)
for idx, row in project_stats.head(10).iterrows():
    proj_name = str(row['Project'])[:40] 
    print(f" -> Project: {proj_name:<40} | Positives: {row['positives']:<5} | Negatives: {row['negatives']:<6} | Total: {row['total']}", flush=True)
print("-" * 60, flush=True)

print("\n[PROGRESS] Standard Random Split (80/20)...", flush=True)

X_tr_raw, X_te_raw, y_tr_raw, y_te_raw = train_test_split(
    X_temp, y_all, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_all
)

print(f"\n[DEBUG] Pre-Balance TRAIN -> Positives: {sum(y_tr_raw==1)} | Negatives: {sum(y_tr_raw==0)}", flush=True)
print(f"[DEBUG] Pre-Balance TEST  -> Positives: {sum(y_te_raw==1)} | Negatives: {sum(y_te_raw==0)}\n", flush=True)

# ==========================================
# 4. TARGET ENCODING (SMELL RISK SCORE)
# ==========================================
if 'Smell' in X_tr_raw.columns:
    print("[PROGRESS] Calculating Smell Risk Score (Target Encoding)...", flush=True)
    
    temp_tr = pd.DataFrame({'Smell': X_tr_raw['Smell'], 'target': y_tr_raw})
    smell_means = temp_tr.groupby('Smell')['target'].mean()
    global_mean = y_tr_raw.mean()
    
    X_tr_raw['Smell_Risk_Score'] = X_tr_raw['Smell'].map(smell_means)
    X_te_raw['Smell_Risk_Score'] = X_te_raw['Smell'].map(smell_means).fillna(global_mean)
    
    X_tr_raw = X_tr_raw.drop(columns=['Smell'])
    X_te_raw = X_te_raw.drop(columns=['Smell'])

if 'Project' in X_tr_raw.columns:
    X_tr_raw = X_tr_raw.drop(columns=['Project'])
    X_te_raw = X_te_raw.drop(columns=['Project'])

X_tr_final = X_tr_raw.select_dtypes(include=[np.number, bool]).astype(np.float32)
X_te_final = X_te_raw.select_dtypes(include=[np.number, bool]).astype(np.float32)

# ==========================================
# 5. BALANCING AND SCALING (MASSIVE SSL PREPARATION)
# ==========================================
print("[PROGRESS] Preparing massive data for SSL and balancing...", flush=True)

scaler = StandardScaler()
X_tr_all_s = scaler.fit_transform(X_tr_final)

def smart_balance(X_p, y_p): 
    pos_idx = y_p[y_p == 1].index
    neg_idx = y_p[y_p == 0].index
    n = min(len(pos_idx), len(neg_idx))
    
    c_pos = np.random.choice(pos_idx, n, replace=False)
    c_neg = np.random.choice(neg_idx, n, replace=False)
    idx = np.concatenate([c_pos, c_neg])
    
    return X_p.loc[idx], y_p.loc[idx]

X_train, y_train = smart_balance(X_tr_final, y_tr_raw)
X_test, y_test = smart_balance(X_te_final, y_te_raw)

X_tr_s = scaler.transform(X_train)
X_te_s = scaler.transform(X_test)

print(f"Massive Dataset (SSL only): {len(X_tr_all_s)} records", flush=True)
print(f"Balanced Dataset - Train: {len(y_train)} | Test: {len(y_test)}", flush=True)

# ==========================================
# 6. HYPERPARAMETER TUNING (XGBoost)
# ==========================================
print("\n--- Searching for best params (XGBoost) ---", flush=True)
param_dist = {
    'max_depth': [3, 4, 6],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.7, 0.9],
    'colsample_bytree': [0.3, 0.5]
}
xgb_search = XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42)

random_search = RandomizedSearchCV(xgb_search, param_distributions=param_dist, n_iter=5, 
                                   cv=3, scoring='f1', n_jobs=-1, random_state=42)

random_search.fit(X_train, y_train)
best_params = random_search.best_params_
print(f"Best params: {best_params}", flush=True)

# ==========================================
# 7. FINAL TRAINING AND NEURAL NETWORKS
# ==========================================
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_train)
X_te_s = scaler.transform(X_test)

models_map = {
    "XGBOOST": XGBClassifier(n_estimators=100, **best_params, random_state=42),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42),
    "LOG_REG": LogisticRegression(max_iter=1000, random_state=42)
}

results_proba = {}

# --- 7.1 Standard Models ---
for name, model in models_map.items():
    print(f"[PROGRESS] Training {name}...", flush=True)
    model.fit(X_tr_s, y_train)
    preds = model.predict(X_te_s)
    results_proba[name] = model.predict_proba(X_te_s)[:, 1]
    print(f"\n>>> {name} REPORT <<<\n", classification_report(y_test, preds), flush=True)

class UnbufferedProgress(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0: 
            print(f" -> [TRAINING] Epoch {epoch+1:03d} | Loss: {logs.get('loss'):.4f} | Val Loss: {logs.get('val_loss'):.4f}", flush=True)

progress_tracker = UnbufferedProgress()
stop_early_strict = callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

# --- 7.2 Baseline Neural Network (Hardcoded) ---
print("\n--- Training Baseline Neural Network (Hardcoded) ---", flush=True)
nn_baseline = models.Sequential([
    layers.Input(shape=(X_tr_s.shape[1],)),
    layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.4), 
    layers.Dense(8, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

nn_baseline.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

nn_baseline.fit(
    X_tr_s, y_train, 
    epochs=100, 
    batch_size=16, 
    validation_split=0.15,
    callbacks=[stop_early_strict, progress_tracker],
    verbose=0 
)

base_probs = nn_baseline.predict(X_te_s, verbose=0).ravel()
base_preds = (base_probs > 0.5).astype(int)
results_proba["NN_BASELINE"] = base_probs

print("\n>>> NN_BASELINE REPORT <<<\n", classification_report(y_test, base_preds), flush=True)

# --- 7.3 Targeted Tuning Neural Network (Bayesian Optimization) ---
print("\n--- Targeted Tuning Neural Network (Bayesian Optimization) ---", flush=True)

def build_targeted_model(hp):
    model = models.Sequential()
    model.add(layers.Input(shape=(X_tr_s.shape[1],)))
    
    hp_units_1 = hp.Int('units_l1', min_value=16, max_value=48, step=8)
    hp_l2_rate = hp.Float('l2_reg', min_value=1e-3, max_value=5e-2, sampling='log')
    model.add(layers.Dense(units=hp_units_1, activation='relu', 
                           kernel_regularizer=regularizers.l2(hp_l2_rate)))
    
    hp_drop_1 = hp.Float('dropout_l1', min_value=0.3, max_value=0.6, step=0.1)
    model.add(layers.Dropout(hp_drop_1))
    
    hp_units_2 = hp.Int('units_l2', min_value=4, max_value=24, step=4)
    model.add(layers.Dense(units=hp_units_2, activation='relu'))
    
    hp_drop_2 = hp.Float('dropout_l2', min_value=0.1, max_value=0.3, step=0.1)
    model.add(layers.Dropout(hp_drop_2))
    
    model.add(layers.Dense(1, activation='sigmoid'))
    
    lr = hp.Float('learning_rate', min_value=5e-4, max_value=5e-3, sampling='log')
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

tuner = kt.BayesianOptimization(
    build_targeted_model,
    objective='val_accuracy',
    max_trials=25,              
    executions_per_trial=2,     
    directory='tuning_dir_targeted',
    project_name='nn_targeted_opt',
    overwrite=True
)

print("[PROGRESS] Running Targeted Bayesian Tuner (Please wait, it will take longer due to double execution)...", flush=True)

search_stop = callbacks.EarlyStopping(monitor='val_loss', patience=8)

tuner.search(X_tr_s, y_train, epochs=60, validation_split=0.15, callbacks=[search_stop], verbose=0)

best_hps_targeted = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
[INFO] Best hyperparameters found by Targeted Tuner:
 - Layer 1 Nodes: {best_hps_targeted.get('units_l1')} (Baseline: 16)
 - Dropout 1: {best_hps_targeted.get('dropout_l1'):.2f} (Baseline: 0.4)
 - Layer 2 Nodes: {best_hps_targeted.get('units_l2')} (Baseline: 8)
 - Dropout 2: {best_hps_targeted.get('dropout_l2'):.2f} (Baseline: 0.2)
 - L2 Reg: {best_hps_targeted.get('l2_reg'):.4f} (Baseline: 0.01)
 - Learning Rate: {best_hps_targeted.get('learning_rate'):.5f}
""", flush=True)

print("[PROGRESS] Final training of Optimized NN...", flush=True)
nn_tuned_targeted = tuner.hypermodel.build(best_hps_targeted)
nn_tuned_targeted.fit(
    X_tr_s, y_train, 
    epochs=100, 
    batch_size=16, 
    validation_split=0.15,
    callbacks=[stop_early_strict, progress_tracker],
    verbose=0
)

tuned_probs = nn_tuned_targeted.predict(X_te_s, verbose=0).ravel()
tuned_preds = (tuned_probs > 0.5).astype(int)
results_proba["NN_TUNED_TARGETED"] = tuned_probs

print("\n>>> NN_TUNED_TARGETED REPORT <<<\n", classification_report(y_test, tuned_preds), flush=True)

# ==========================================
# 7.5 MASKED AUTOENCODER PRE-TRAINING (TABULAR SSL)
# ==========================================
print("\n--- Pre-Training Self-Supervised Masked Autoencoder ---", flush=True)

def apply_swap_noise_and_mask(X, swap_prob=0.15, mask_prob=0.15):
    X_corrupted = X.copy()
    n_samples, n_features = X.shape
    
    for i in range(n_features):
        mask = np.random.rand(n_samples) < swap_prob
        swap_idx = np.random.permutation(n_samples)
        X_corrupted[mask, i] = X[swap_idx[mask], i]
        
    mask_zero = np.random.rand(n_samples, n_features) < mask_prob
    X_corrupted[mask_zero] = 0.0
    
    return X_corrupted

print("[PROGRESS] Creating corrupted views (Swap Noise + Masking)...", flush=True)
X_tr_corrupted = apply_swap_noise_and_mask(X_tr_s, swap_prob=0.15, mask_prob=0.15)

input_dim = X_tr_s.shape[1]
ae_input = layers.Input(shape=(input_dim,))

encoded = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(ae_input)
encoded = layers.Dropout(0.3)(encoded)
encoded = layers.Dense(16, activation='relu')(encoded)

decoded = layers.Dense(32, activation='relu')(encoded)
decoded = layers.Dense(input_dim, activation='linear')(decoded)

autoencoder = models.Model(ae_input, decoded)
encoder = models.Model(ae_input, encoded, name="ssl_encoder")

autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse')

class UnbufferedProgressSSL(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            print(f" -> [SSL PRE-TRAIN] Epoch {epoch+1:03d} | Loss (MSE): {logs.get('loss'):.4f} | Val Loss: {logs.get('val_loss'):.4f}", flush=True)

print("[PROGRESS] Training Autoencoder (learning to reconstruct clean original data)...", flush=True)
autoencoder.fit(
    X_tr_corrupted, X_tr_s,
    epochs=60,
    batch_size=32,
    validation_split=0.15,
    callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True), UnbufferedProgressSSL()],
    verbose=0
)

print("\n[PROGRESS] Building classifier for Fine-tuning...", flush=True)

clf_input = layers.Input(shape=(input_dim,))
ssl_features = encoder(clf_input)
x = layers.Dropout(0.2)(ssl_features)
x = layers.Dense(8, activation='relu')(x)
clf_output = layers.Dense(1, activation='sigmoid')(x)

ssl_classifier = models.Model(clf_input, clf_output)

class UnbufferedProgressFT(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 5 == 0:
            print(f" -> [SSL FINE-TUNE] Epoch {epoch+1:03d} | Loss: {logs.get('loss'):.4f} | Val Acc: {logs.get('val_accuracy'):.4f}", flush=True)

tracker_ft = UnbufferedProgressFT()

# --- Phase 1: Freeze ---
print("[PROGRESS] Phase 1: Freeze Encoder (Warm-up of classification head)...", flush=True)
encoder.trainable = False 
ssl_classifier.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                       loss='binary_crossentropy', metrics=['accuracy'])

ssl_classifier.fit(
    X_tr_s, y_train,
    epochs=30, batch_size=16, validation_split=0.15,
    callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True), tracker_ft],
    verbose=0
)

# --- Phase 2: Thaw ---
print("[PROGRESS] Phase 2: Thaw Encoder (Fine-tuning end-to-end)...", flush=True)
encoder.trainable = True
ssl_classifier.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                       loss='binary_crossentropy', metrics=['accuracy'])

ssl_classifier.fit(
    X_tr_s, y_train,
    epochs=40, batch_size=16, validation_split=0.15,
    callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True), tracker_ft],
    verbose=0
)

ssl_probs = ssl_classifier.predict(X_te_s, verbose=0).ravel()
ssl_preds = (ssl_probs > 0.5).astype(int)

results_proba["SSL_AUTOENCODER"] = ssl_probs
print("\n>>> SSL_AUTOENCODER REPORT <<<\n", classification_report(y_test, ssl_preds), flush=True)

# ==========================================
# 7.6 MASSIVE CONTRASTIVE LEARNING (SSL)
# ==========================================
print("\n--- Pre-Training Self-Supervised Contrastive (Massive Data) ---", flush=True)

def augment_view(X, swap_prob=0.15):
    X_aug = X.copy()
    n_samples, n_features = X.shape
    for i in range(n_features):
        mask = np.random.rand(n_samples) < swap_prob
        swap_idx = np.random.permutation(n_samples)
        X_aug[mask, i] = X[swap_idx[mask], i]
    return X_aug

def generate_massive_pairs(X, num_pairs):
    num_samples = len(X)
    idx1 = np.random.randint(0, num_samples, num_pairs)
    idx2 = np.random.randint(0, num_samples, num_pairs)
    
    half = num_pairs // 2
    idx2[:half] = idx1[:half]
    
    labels = np.zeros(num_pairs)
    labels[:half] = 1 
    
    X1 = augment_view(X[idx1])
    X2 = augment_view(X[idx2])
    
    shuf_idx = np.random.permutation(num_pairs)
    return X1[shuf_idx], X2[shuf_idx], labels[shuf_idx]

print(f"[PROGRESS] Generating contrastive pairs from massive dataset...", flush=True)

n_pairs = len(X_tr_all_s) * 2
X1_mass, X2_mass, labels_mass = generate_massive_pairs(X_tr_all_s, num_pairs=n_pairs)

input_dim = X_tr_all_s.shape[1]
encoder_input = layers.Input(shape=(input_dim,))

encoded = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(encoder_input)
encoded = layers.Dropout(0.3)(encoded)
encoded = layers.Dense(16, activation='relu')(encoded)
encoded = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(encoded)

encoder_contrastive = models.Model(encoder_input, encoded, name="encoder_contrastive")

input_a = layers.Input(shape=(input_dim,))
input_b = layers.Input(shape=(input_dim,))

emb_a = encoder_contrastive(input_a)
emb_b = encoder_contrastive(input_b)

cos_sim = layers.Dot(axes=1, normalize=False)([emb_a, emb_b])
siamese_massive = models.Model([input_a, input_b], cos_sim)

def contrastive_loss_cosine(margin=0.2):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        pos_loss = y_true * (1.0 - y_pred)
        neg_loss = (1.0 - y_true) * tf.maximum(0.0, y_pred - margin)
        return tf.reduce_mean(pos_loss + neg_loss)
    return loss

siamese_massive.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                        loss=contrastive_loss_cosine(margin=0.1))

class UnbufferedProgressContrastive(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 5 == 0:
            print(f" -> [CONTRASTIVE PRE-TRAIN] Epoch {epoch+1:03d} | Loss: {logs.get('loss'):.4f} | Val Loss: {logs.get('val_loss'):.4f}", flush=True)

print(f"[PROGRESS] Training Siamese Network on {n_pairs} pairs...", flush=True)
siamese_massive.fit(
    [X1_mass, X2_mass], labels_mass,
    epochs=40, batch_size=64, validation_split=0.15,
    callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True), UnbufferedProgressContrastive()],
    verbose=0
)

print("\n[PROGRESS] Fine-tuning Contrastive Encoder on BALANCED dataset...", flush=True)

clf_input = layers.Input(shape=(input_dim,))
ssl_features = encoder_contrastive(clf_input)
x = layers.Dropout(0.2)(ssl_features)
x = layers.Dense(8, activation='relu')(x)
clf_output = layers.Dense(1, activation='sigmoid')(x)

contrastive_classifier = models.Model(clf_input, clf_output)

class UnbufferedProgressFT_Contr(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            print(f" -> [CONTRASTIVE FINE-TUNE] Epoch {epoch+1:03d} | Loss: {logs.get('loss'):.4f} | Val Acc: {logs.get('val_accuracy'):.4f}", flush=True)

tracker_ft_contr = UnbufferedProgressFT_Contr()

# --- Phase 1: Freeze ---
encoder_contrastive.trainable = False 
contrastive_classifier.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
                               loss='binary_crossentropy', metrics=['accuracy'])

contrastive_classifier.fit(
    X_tr_s, y_train, 
    epochs=30, batch_size=16, validation_split=0.15,
    callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True), tracker_ft_contr],
    verbose=0
)

# --- Phase 2: Thaw ---
encoder_contrastive.trainable = True
contrastive_classifier.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                               loss='binary_crossentropy', metrics=['accuracy'])

contrastive_classifier.fit(
    X_tr_s, y_train, 
    epochs=40, batch_size=16, validation_split=0.15,
    callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True), tracker_ft_contr],
    verbose=0
)

contr_probs = contrastive_classifier.predict(X_te_s, verbose=0).ravel()
contr_preds = (contr_probs > 0.5).astype(int)

results_proba["SSL_MASSIVE_CONTRASTIVE"] = contr_probs
print("\n>>> SSL_MASSIVE_CONTRASTIVE REPORT <<<\n", classification_report(y_test, contr_preds), flush=True)

# ==========================================
# 8. ABLATION STUDY (REMOVING EMBEDDINGS)
# ==========================================
print("\n--- Starting Ablation Study (Removing 'emb_' columns) ---", flush=True)

# 1. Identify and remove embedding columns
emb_cols = [c for c in X_train.columns if "emb_" in c]
classic_cols = [c for c in X_train.columns if "emb_" not in c]

print(f"[INFO] Total columns: {len(X_train.columns)}", flush=True)
print(f"[INFO] Embedding columns removed: {len(emb_cols)}", flush=True)
print(f"[INFO] Classic columns kept: {len(classic_cols)}", flush=True)

# 2. Prepare ablated datasets
X_train_abl = X_train[classic_cols]
X_test_abl = X_test[classic_cols]

scaler_abl = StandardScaler()
X_tr_abl_s = scaler_abl.fit_transform(X_train_abl)
X_te_abl_s = scaler_abl.transform(X_test_abl)

# --- 8.1 Tuning e Training XGBoost (Ablated) ---
print("\n[PROGRESS] Searching for best params for XGBoost (Ablated)...", flush=True)
param_dist_abl = {
    'max_depth': [3, 4, 6],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.7, 0.9],
    'colsample_bytree': [0.3, 0.5, 0.8] 
}

xgb_search_abl = XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42)
random_search_abl = RandomizedSearchCV(xgb_search_abl, param_distributions=param_dist_abl, n_iter=5, 
                                       cv=3, scoring='f1', n_jobs=-1, random_state=42)

random_search_abl.fit(X_tr_abl_s, y_train)
best_params_abl = random_search_abl.best_params_
print(f"[INFO] Best params (Ablated): {best_params_abl}", flush=True)

print("[PROGRESS] Training XGBoost (Ablated)...", flush=True)
xgb_abl = XGBClassifier(n_estimators=100, **best_params_abl, random_state=42)
xgb_abl.fit(X_tr_abl_s, y_train)

xgb_abl_probs = xgb_abl.predict_proba(X_te_abl_s)[:, 1]
xgb_abl_preds = xgb_abl.predict(X_te_abl_s)
results_proba["XGB_ABLATED"] = xgb_abl_probs

print("\n>>> XGB_ABLATED REPORT <<<\n", classification_report(y_test, xgb_abl_preds), flush=True)

# --- 8.2 Baseline NN Ablated ---
print("[PROGRESS] Training Baseline NN (Ablated)...", flush=True)
nn_abl = models.Sequential([
    layers.Input(shape=(X_tr_abl_s.shape[1],)),
    layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.4), 
    layers.Dense(8, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

nn_abl.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

nn_abl.fit(
    X_tr_abl_s, y_train, 
    epochs=100, 
    batch_size=16, 
    validation_split=0.15,
    callbacks=[stop_early_strict, progress_tracker], 
    verbose=0 
)

nn_abl_probs = nn_abl.predict(X_te_abl_s, verbose=0).ravel()
nn_abl_preds = (nn_abl_probs > 0.5).astype(int)
results_proba["NN_ABLATED"] = nn_abl_probs

print("\n>>> NN_ABLATED REPORT <<<\n", classification_report(y_test, nn_abl_preds), flush=True)

# ==========================================
# 8.5 STACKING ENSEMBLE (LATE FUSION)
# ==========================================
print("\n--- Starting Stacking Ensemble (Late Fusion) ---", flush=True)

from sklearn.model_selection import KFold

X_train_classiche = X_train[classic_cols].values
X_train_embeddings = X_train[emb_cols].values
X_test_classiche = X_test[classic_cols].values
X_test_embeddings = X_test[emb_cols].values

scaler_class = StandardScaler()
X_tr_class_s = scaler_class.fit_transform(X_train_classiche)
X_te_class_s = scaler_class.transform(X_test_classiche)

scaler_emb = StandardScaler()
X_tr_emb_s = scaler_emb.fit_transform(X_train_embeddings)
X_te_emb_s = scaler_emb.transform(X_test_embeddings)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds_xgb = np.zeros(len(y_train))
oof_preds_nn = np.zeros(len(y_train))

print("[PROGRESS] Generating OOF predictions via 5-Fold CV...", flush=True)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_tr_class_s)):
    print(f" -> Processing Fold {fold+1}/5...", flush=True)
    
    X_tr_c, X_val_c = X_tr_class_s[train_idx], X_tr_class_s[val_idx]
    X_tr_e, X_val_e = X_tr_emb_s[train_idx], X_tr_emb_s[val_idx]
    y_tr, y_val = y_train.values[train_idx], y_train.values[val_idx]
    
    m_xgb = XGBClassifier(n_estimators=100, **best_params_abl, random_state=42)
    m_xgb.fit(X_tr_c, y_tr)
    oof_preds_xgb[val_idx] = m_xgb.predict_proba(X_val_c)[:, 1]
    
    m_nn = models.Sequential([
        layers.Input(shape=(X_tr_e.shape[1],)),
        layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.4),
        layers.Dense(8, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    m_nn.compile(optimizer='adam', loss='binary_crossentropy')
    
    # Added validation_data to fix EarlyStopping warning
    m_nn.fit(X_tr_e, y_tr, epochs=50, batch_size=16, verbose=0, 
             validation_data=(X_val_e, y_val),
             callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])
    oof_preds_nn[val_idx] = m_nn.predict(X_val_e, verbose=0).ravel()

X_meta_train = np.column_stack((oof_preds_xgb, oof_preds_nn))
meta_learner = LogisticRegression()
meta_learner.fit(X_meta_train, y_train)

weights = meta_learner.coef_[0]
print(f"[INFO] Meta-Learner Weights -> XGB (Classic): {weights[0]:.4f}, NN (Embeddings): {weights[1]:.4f}", flush=True)

# --- Final Inference ---
print("[PROGRESS] Final Stacking Inference...", flush=True)

xgb_expert_final = XGBClassifier(n_estimators=100, **best_params_abl, random_state=42)
xgb_expert_final.fit(X_tr_class_s, y_train)

nn_expert_final = models.Sequential([
    layers.Input(shape=(X_tr_emb_s.shape[1],)), # Specifically expects 64
    layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.4),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
nn_expert_final.compile(optimizer='adam', loss='binary_crossentropy')
nn_expert_final.fit(X_tr_emb_s, y_train, epochs=50, batch_size=16, verbose=0)

test_preds_xgb = xgb_expert_final.predict_proba(X_te_class_s)[:, 1]
# FIXED: Changed X_te_s to X_te_emb_s (the 64 columns only)
test_preds_nn = nn_expert_final.predict(X_te_emb_s, verbose=0).ravel() 

X_meta_test = np.column_stack((test_preds_xgb, test_preds_nn))
stacking_probs = meta_learner.predict_proba(X_meta_test)[:, 1]
stacking_preds = (stacking_probs > 0.5).astype(int)

results_proba["STACKING_ENSEMBLE"] = stacking_probs

print("\n>>> STACKING_ENSEMBLE REPORT <<<\n", classification_report(y_test, stacking_preds), flush=True)


# ==========================================
# 8.7 PCA DIMENSIONALITY REDUCTION + XGBOOST
# ==========================================
print("\n--- Starting PCA Reduction + XGBoost ---", flush=True)

from sklearn.decomposition import PCA

# 1. Applichiamo la PCA solo sulle 64 colonne di embedding
# Estraiamo 3 componenti che solitamente catturano l'essenza semantica
n_pca = 3
pca = PCA(n_components=n_pca, random_state=42)

print(f"[PROGRESS] Reducing {len(emb_cols)} embeddings to {n_pca} PCA components...", flush=True)
train_pca = pca.fit_transform(X_tr_emb_s)
test_pca = pca.transform(X_te_emb_s)

# Creiamo i nomi delle nuove colonne
pca_cols = [f"pca_emb_{i}" for i in range(n_pca)]
train_pca_df = pd.DataFrame(train_pca, columns=pca_cols)
test_pca_df = pd.DataFrame(test_pca, columns=pca_cols)

# 2. Fondiamo le metriche classiche (Ablated) con le nuove componenti PCA
# Usiamo i dati già scalati delle metriche classiche per coerenza
X_tr_pca_combined = np.hstack((X_tr_class_s, train_pca))
X_te_pca_combined = np.hstack((X_te_class_s, test_pca))

print(f"[INFO] New feature count: {X_tr_pca_combined.shape[1]} (Classic + {n_pca} PCA)", flush=True)
print(f"[INFO] Explained variance ratio by PCA: {np.sum(pca.explained_variance_ratio_):.4f}", flush=True)

# 3. Tuning rapido e Training di XGBoost su questo set ibrido "pulito"
print("[PROGRESS] Training XGBoost on Classic + PCA features...", flush=True)

xgb_pca = XGBClassifier(n_estimators=100, **best_params_abl, random_state=42)
xgb_pca.fit(X_tr_pca_combined, y_train)

xgb_pca_probs = xgb_pca.predict_proba(X_te_pca_combined)[:, 1]
xgb_pca_preds = (xgb_pca_probs > 0.5).astype(int)

results_proba["XGB_WITH_PCA_EMB"] = xgb_pca_probs

print("\n>>> XGB_WITH_PCA_EMB REPORT <<<\n", classification_report(y_test, xgb_pca_preds), flush=True)

# ==========================================
# 9. PLOTS GENERATION
# ==========================================
print("\n--- Generating Plots ---", flush=True)

# --- ROC ---
plt.figure(figsize=(8,6))
for name, proba in results_proba.items():
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.2f})")
plt.plot([0,1],[0,1], 'k--')
plt.legend()
plt.savefig(os.path.join(script_dir, "roc_comparison.png"))
plt.close()

# --- Feature Importance (Original XGBoost) ---
plt.figure(figsize=(10,8))
imp = models_map["XGBOOST"].feature_importances_
idx = np.argsort(imp)[::-1][:20]
plt.barh(range(20), imp[idx], align='center', color='teal')
plt.yticks(range(20), [X_train.columns[i] for i in idx])
plt.gca().invert_yaxis()
plt.savefig(os.path.join(script_dir, "feature_importance.png"), bbox_inches='tight')
plt.close()

# --- UMAP ---
print("[PROGRESS] Generating UMAP...", flush=True)
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
emb = reducer.fit_transform(X_te_s)
plt.figure(figsize=(10,8))
plt.scatter(emb[:, 0], emb[:, 1], c=y_test, cmap='coolwarm', s=10)
plt.savefig(os.path.join(script_dir, "umap_test.png"))
plt.close()

# --- SHAP (Original XGBoost) ---
print("[PROGRESS] Computing SHAP values...", flush=True)
explainer = shap.TreeExplainer(models_map["XGBOOST"])
shap_vals = explainer.shap_values(pd.DataFrame(X_te_s, columns=X_train.columns))
plt.figure()
shap.summary_plot(shap_vals, pd.DataFrame(X_te_s, columns=X_train.columns), show=False)
plt.savefig(os.path.join(script_dir, "shap_summary.png"), bbox_inches='tight')
plt.close()

print(f"\n[DONE] Script done. Work saved in: {script_dir}", flush=True)