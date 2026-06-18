# Predicting Refactoring Interventions via Hybrid Feature Spaces

This repository contains the replication package and pipeline scripts for a comprehensive empirical study bridging traditional static code analysis (object-oriented metrics) with advanced graph representation learning (embeddings of Control Flow Graphs) to predict refactoring interventions on smelly methods.

---

## 1. Methodology Overview

The methodology combines traditional software engineering metrics with dense, vector-based representations of Control Flow Graphs (CFGs) for predicting specific refactoring techniques on code-smelly methods. 

```mermaid
flowchart TD
    subgraph Step 1: Initial Metrics & History
        A[Java Repositories] -->|RefactoringMiner| B[Refactoring JSON]
        A -->|DesigniteJava Snapshots| C[Smells CSVs]
        A -->|run_ck_analysis.py| D[CK OO Metrics]
    end

    subgraph Step 2: Mining & Labeling
        B & C -->|run_miner.py & run_evolution.py| E[Tracked Movements & Resolved Smells]
        E -->|run_labeller.py| F[Supervised Labeled Dataset]
    end

    subgraph Step 3: Graph Embedding
        A -->|wsl_pipeline.py / test.py| G[64D WSL CFG Embeddings]
        A -->|GraphCode2Vec| GK[GraphCode2Vec Embeddings]
    end

    subgraph Step 4: Dataset Consolidation
        F & D -->|run_ck_merger.py| H[Merged Labeled & CK Metrics]
        H & G -->|run_all_enrichments.py| I[Consolidated Dataset (WSL)]
        H & GK -->|merge_graphcode2vec_ck.py| I2[Consolidated Dataset (GraphCode2Vec)]
    end

    subgraph Step 5: Machine Learning
        I & I2 -->|models_pipeline/run_all.py| J[Logistic Regression, SVM, XGBoost, DNN, Late Fusion Stacking]
    end
```

---

## 2. Directory Structure

The codebase is organized into modular directories under `src/`:

```
project_thesis/
├── README.md
└── src/
    ├── core/               # Core execution scripts and processing logic
    ├── runners/            # Batch runners and automation scripts
    ├── utils/              # General helper and utility scripts
    └── models_pipeline/    # Machine learning training, tuning, and evaluation
```

### Core Execution Scripts ([src/core/](file:///D:/papersEvolution/DesigniteJava/project_thesis/src/core))
- **[miner.py](file:///D:/papersEvolution/DesigniteJava/project_thesis/src/core/miner.py)**: Chronologically tracks class renames and movements across commits and maps refactoring interventions to smelly methods.
- **[evolution.py](file:///D:/papersEvolution/DesigniteJava/project_thesis/src/core/evolution.py)**: Compares the initial and final snapshots and checks if smelly methods are resolved or persisted (smells still present in the "after" snapshot or refactored signature).
- **[labeller.py](file:///D:/papersEvolution/DesigniteJava/project_thesis/src/core/labeller.py)**: Appends supervised labels (binary `0` or `1` fields) representing 10 refactoring categories to the Designite metrics dataset.
- **[ck_merger.py](file:///D:/papersEvolution/DesigniteJava/project_thesis/src/core/ck_merger.py)**: Cleans and matches Designite methods with CK class-level and method-level metrics reports.
- **[enrich_embeddings.py](file:///D:/papersEvolution/DesigniteJava/project_thesis/src/core/enrich_embeddings.py)**: Fuzzy-merges datasets with CFG embeddings based on normalized class names and line number proximity.
- **[build_dataset.py](file:///D:/papersEvolution/DesigniteJava/project_thesis/src/core/build_dataset.py)**: Scans Java files and integrates CodeBERT embeddings if enabled.
- **[merger.py](file:///D:/papersEvolution/DesigniteJava/project_thesis/src/core/merger.py)**: Combines labeled datasets and separates positive instances (rows with refactorings) from negative instances (zeros).
- **[merge_graphcode2vec_ck.py](file:///D:/papersEvolution/DesigniteJava/project_thesis/src/core/merge_graphcode2vec_ck.py)**: Merges generated GraphCode2Vec embeddings with CK metrics.

### Utility Helpers ([src/utils/](file:///D:/papersEvolution/DesigniteJava/project_thesis/src/utils))
- **[archiver.py](file:///D:/papersEvolution/DesigniteJava/project_thesis/src/utils/archiver.py)** / **[archiver_after.py](file:///D:/papersEvolution/DesigniteJava/project_thesis/src/utils/archiver_after.py)**: Automate moving raw complex methods reports from individual project directories to backup archive directories.
- **[ck_onefile.py](file:///D:/papersEvolution/DesigniteJava/project_thesis/src/utils/ck_onefile.py)**: Merges all individual project CK files into a single master output CSV.
- **[counter.py](file:///D:/papersEvolution/DesigniteJava/project_thesis/src/utils/counter.py)**: Utility to analyze and output refactoring occurrences grouped by classes.
- **[meanPooling.py](file:///D:/papersEvolution/DesigniteJava/project_thesis/src/utils/meanPooling.py)**: Aggregates node-level embeddings to obtain method-level vectors.
- **[megaList.py](file:///D:/papersEvolution/DesigniteJava/project_thesis/src/utils/megaList.py)**: Combines lists or lines from different files into a consolidated collection.
- **[merge_csv.py](file:///D:/papersEvolution/DesigniteJava/project_thesis/src/utils/merge_csv.py)**: Simple helper to concatenate list-based csv data.
- **[IDConverter.py](file:///D:/papersEvolution/DesigniteJava/project_thesis/src/utils/IDConverter.py)**: Map and convert string-based node IDs in global edgelists to integers.
- **[features_classes.py](file:///D:/papersEvolution/DesigniteJava/project_thesis/src/utils/features_classes.py)**, **[features_methods.py](file:///D:/papersEvolution/DesigniteJava/project_thesis/src/utils/features_methods.py)**, **[features_smells.py](file:///D:/papersEvolution/DesigniteJava/project_thesis/src/utils/features_smells.py)**, **[load_data.py](file:///D:/papersEvolution/DesigniteJava/project_thesis/src/utils/load_data.py)**, **[merge_features.py](file:///D:/papersEvolution/DesigniteJava/project_thesis/src/utils/merge_features.py)**: Sub-modular feature engineering helpers used during initial dataset construction in `build_dataset.py`.

### Batch Runner & Automation Wrappers ([src/runners/](file:///D:/papersEvolution/DesigniteJava/project_thesis/src/runners))
- **[run_ck_analysis.py](file:///D:/papersEvolution/DesigniteJava/project_thesis/src/runners/run_ck_analysis.py)**: Batch runs the CK tool jar over the projects to generate raw OO metrics reports.
- **[run_miner.py](file:///D:/papersEvolution/DesigniteJava/project_thesis/src/runners/run_miner.py)**: Batch wrapper running `miner.py` across all project targets.
- **[run_evolution.py](file:///D:/papersEvolution/DesigniteJava/project_thesis/src/runners/run_evolution.py)**: Batch wrapper running `evolution.py` to compare before and after states.
- **[run_labeller.py](file:///D:/papersEvolution/DesigniteJava/project_thesis/src/runners/run_labeller.py)**: Batch wrapper executing the labeling process using `labeller.py`.
- **[run_ck_merger.py](file:///D:/papersEvolution/DesigniteJava/project_thesis/src/runners/run_ck_merger.py)**: Batch wrapper merging labeled features with CK metrics datasets.
- **[run_all_enrichments.py](file:///D:/papersEvolution/DesigniteJava/project_thesis/src/runners/run_all_enrichments.py)**: Batch wrapper integrating CFG embeddings into the merged OO metric tables.
- **[wsl_pipeline.py](file:///D:/papersEvolution/DesigniteJava/project_thesis/src/runners/wsl_pipeline.py)** / **[test.py](file:///D:/papersEvolution/DesigniteJava/project_thesis/src/runners/test.py)**: Automates the entire Linux/WSL execution flow (Joern parsing, CFG dot export, Scala metadata extraction, edge conversion, LINE node embedding training, and mean-pooling).
- **[run_designite_final.py](file:///D:/papersEvolution/DesigniteJava/project_thesis/src/runners/run_designite_final.py)**: Automation script running the DesigniteJava tool jar to extract OO and code smell metrics.
- **[run_build_dataset_before.py](file:///D:/papersEvolution/DesigniteJava/project_thesis/src/runners/run_build_dataset_before.py)** / **[run_build_dataset_after.py](file:///D:/papersEvolution/DesigniteJava/project_thesis/src/runners/run_build_dataset_after.py)**: Batch wrappers running the initial metric dataset build pipeline.


### Machine Learning Pipeline ([src/models_pipeline/](file:///D:/papersEvolution/DesigniteJava/project_thesis/src/models_pipeline))
- **[pipeline.py](file:///D:/papersEvolution/DesigniteJava/project_thesis/src/models_pipeline/pipeline.py)**: The core machine learning execution script that scales/balances datasets, trains LR/SVM/XGB/DNN models, performs Late Fusion stacking, runs ablation studies, applies Self-Supervised Learning (SSL), and plots calibration, ROC, UMAP, and SHAP outputs.
- **[run_all.py](file:///D:/papersEvolution/DesigniteJava/project_thesis/src/models_pipeline/run_all.py)**: Batch script executing `pipeline.py` sequentially across all 10 target refactorings.

---

## 3. Recommended Execution Order

All runner scripts dynamically resolve paths relative to their locations, making them executable from any directory. To reproduce the study's data acquisition, features consolidation, and model evaluation, execute the scripts in the following order:

### Step 1: Initial Metrics and Refactoring Extraction
Extract structural metrics from snapshots and historical refactorings from commits:
1. **Run DesigniteJava**: Run on the *before* and *after* project versions (can be automated using `python src/runners/run_designite_final.py`).
2. **Run RefactoringMiner**: Extract JSON history files from your project repositories.
3. **Run CK Metrics**: Extract 54 object-oriented metrics:
   ```bash
   python src/runners/run_ck_analysis.py
   ```

### Step 2: Mining and Labeling (Phase 1)
Align the smells with evolutionary refactoring events and generate targets:
1. **Mine Refactorings**: Align commit diffs with smelly methods:
   ```bash
   python src/runners/run_miner.py
   ```
2. **Evaluate State Evolution**: Check if smells were resolved or persisted:
   ```bash
   python src/runners/run_evolution.py
   ```
3. **Apply Labels**: Generate binary columns representing refactoring techniques:
   ```bash
   python src/runners/run_labeller.py
   ```

### Step 3: Graph Embedding Generation
Generate the method-level CFG/Graph embeddings. You can choose one of the following two options depending on your setup:

#### Option A: WSL/Linux CFG Pipeline
Runs Joern CFG parsing, Scala metadata dump, edge conversion, LINE node-embedding training, and mean-pooling:
1. **Execute WSL/Linux Script**:
   ```bash
   # Execute in a Linux or WSL environment with Joern installed
   python src/runners/wsl_pipeline.py
   ```

#### Option B: GraphCode2Vec Embeddings
Alternatively, you can generate graph embeddings using GraphCode2Vec.

---

### Step 4: Dataset Consolidation (Phase 2 & 3)
Fuse metrics and embeddings together, depending on the embedding option selected:

#### For Option A (WSL/Linux Pipeline Embeddings)
1. **Merge Labeled Data with CK Metrics**:
   ```bash
   python src/runners/run_ck_merger.py
   ```
2. **Integrate CFG Embeddings**: Merge the embeddings by performing a fuzzy join based on line number proximity:
   ```bash
   python src/runners/run_all_enrichments.py
   ```

#### For Option B (GraphCode2Vec Embeddings)
1. **Merge Labeled Data with CK Metrics**:
   ```bash
   python src/runners/run_ck_merger.py
   ```
2. **Integrate GraphCode2Vec Embeddings**: Run the dedicated [merge_graphcode2vec_ck.py](file:///D:/papersEvolution/DesigniteJava/project_thesis/src/core/merge_graphcode2vec_ck.py) script to merge the GraphCode2Vec embeddings with the CK metrics:
   ```bash
   python src/core/merge_graphcode2vec_ck.py --ck_file <path_to_ck_merged_csv> --embeddings_file <path_to_graphcode2vec_csv> --output_file <path_to_output_csv>
   ```

### Step 5: Machine Learning Execution
Train and evaluate the models:
1. **Run ML Pipelines**:
   ```bash
   python src/models_pipeline/run_all.py
   ```
   *This trains Logistic Regression, SVM, XGBoost, and Deep Neural Networks (with Bayesian Tuning), evaluates Stacking Late Fusion, performs ablation studies, and generates plots (ROC, Calibration, UMAP, SHAP) in the `classification_report` output folder.*
