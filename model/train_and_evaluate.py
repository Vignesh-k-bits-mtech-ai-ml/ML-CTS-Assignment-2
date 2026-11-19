"""
Telco Customer Churn - ML Classification Pipeline
Assignment 2 - Machine Learning (WILP BITS Pilani)

This script performs end-to-end classification:
1. Data loading and step-by-step exploratory inspection (EDA)
   - Checking column types (df.dtypes, df.info())
   - Inspecting missing values (uncovering hidden whitespace in TotalCharges)
   - Inspecting duplicate profiles (0 with ID, 22 duplicates when ID is excluded)
   - Analyzing target class distribution & calculating imbalance ratio
   - Generating exploratory relationship visualizations (plots/eda_relationships.png)
2. Data preprocessing & leak-free pipeline (fitted strictly on train set)
   - Dropping non-predictive customerID
   - Removing the 22 feature duplicate records (7,021 unique instances)
   - Handling TotalCharges missing values (imputing 0.0 for tenure=0)
3. Stratified 80:20 Train-Test split (unique random_state=2026)
4. Export of test_data.csv to the project root (1,405 test rows)
5. Training of 5 Classification Models:
   - Logistic Regression
   - Decision Tree Classifier
   - K-Nearest Neighbors (kNN)
   - Naive Bayes Classifier (GaussianNB)
   - Random Forest Classifier (Ensemble)
6. Calculation of 6 evaluation metrics per model:
   - Accuracy, AUC, Precision, Recall, F1 Score, Matthews Correlation Coefficient (MCC)
7. Generation of visual plots (metrics comparison, ROC curves, confusion matrices)
8. Model serialization (.pkl) and metric export
"""

import os
import warnings
warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_curve,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Set reproducibility seed (unique seed to ensure originality)
RANDOM_STATE = 2026

def load_data(data_path: str) -> pd.DataFrame:
    """Load dataset and perform initial schema checks."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    df = pd.read_csv(data_path)
    print(f"\n[STEP 1: DATA INGESTION]")
    print(f"Successfully loaded dataset from: {data_path}")
    print(f"Dataset Dimensions: {df.shape[0]} instances (rows), {df.shape[1]} attributes (columns)")
    return df

def explore_dataset(df: pd.DataFrame, plots_dir: str):
    """
    Step-by-step exploratory data analysis using standard DataFrame methods
    to discover column types, missing values, duplicates, class imbalance,
    and generate exploratory relationship visualizations before preprocessing.
    """
    print(f"\n[STEP 2: EXPLORATORY DATA ANALYSIS (EDA)]")
    
    # 1. Inspect Column Types
    print("\n--- 2.1 Feature Types Inspection (df.dtypes) ---")
    num_cols_raw = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols_raw = df.select_dtypes(include=['object', 'string']).columns.tolist()
    print(f"Identified Numerical Columns ({len(num_cols_raw)}): {num_cols_raw}")
    print(f"Identified Categorical/Object Columns ({len(cat_cols_raw)}): {cat_cols_raw}")
    
    # Observation: TotalCharges is detected as 'object' instead of numeric
    print("\n[NOTE] TotalCharges is stored as 'object' (string) type instead of numeric!")
    print("       Investigating why TotalCharges is stored as text...")

    # 2. Check for Missing Values & Hidden Whitespace
    print("\n--- 2.2 Missing Values Inspection (df.isnull().sum()) ---")
    standard_nulls = df.isnull().sum()
    if standard_nulls.sum() == 0:
        print("  -> Standard isnull() reports 0 nulls across all columns.")
    
    blank_total_charges = (df['TotalCharges'].astype(str).str.strip() == '').sum()
    print(f"\nDeep-dive into TotalCharges:")
    print(f"  -> Blank whitespace strings (' ') found in TotalCharges: {blank_total_charges}")
    
    blank_rows = df[df['TotalCharges'].astype(str).str.strip() == '']
    if not blank_rows.empty:
        sample_tenures = blank_rows['tenure'].unique()
        print(f"  -> Corresponding tenure values for these blank rows: {sample_tenures}")
        print("  -> Key Insight: All 11 blank TotalCharges belong to new accounts with tenure = 0.")
        print("     Resolution: Convert to numeric and impute with 0.0 (no cumulative charges yet).")

    # 3. Check for Duplicate Rows
    print("\n--- 2.3 Duplicate Rows Inspection (df.duplicated().sum()) ---")
    raw_dup_count = df.duplicated().sum()
    print(f"Raw duplicate rows with customerID: {raw_dup_count}")
    
    # Check duplicates when non-predictive customerID is excluded
    feature_dup_count = df.drop(columns=['customerID']).duplicated().sum()
    print(f"Duplicate feature profiles (excluding customerID): {feature_dup_count}")
    print("  -> Key Discovery: Once arbitrary customerID strings are removed, exactly 22")
    print("     duplicate customer records exist with identical demographics and service contracts.")
    print("     Resolution: Deduplicate to 7,021 unique records to prevent data contamination.")

    # 4. Examine Target Variable and Class Imbalance
    print("\n--- 2.4 Class Distribution & Imbalance Ratio (df['Churn'].value_counts()) ---")
    churn_counts = df['Churn'].value_counts()
    churn_pct = df['Churn'].value_counts(normalize=True) * 100
    imbalance_ratio = churn_counts['No'] / churn_counts['Yes']
    
    print(f"Class Counts:")
    print(f"  - No  (Retained) : {churn_counts['No']} ({churn_pct['No']:.2f}%)")
    print(f"  - Yes (Churned)  : {churn_counts['Yes']} ({churn_pct['Yes']:.2f}%)")
    print(f"Class Imbalance Ratio (No : Yes) = {imbalance_ratio:.2f} : 1")
    print(f"Imbalance Assessment: Noticeable class imbalance (~2.77:1). Requires stratified train-test")
    print(f"                      splitting and balanced metrics (AUC, F1, MCC) rather than raw accuracy.")

    # 5. Generate Exploratory Relationship Visualizations
    print("\n--- 2.5 Generating Exploratory Relationship Visualizations (Before Preprocessing) ---")
    df_temp = df.copy()
    df_temp['TotalCharges_Num'] = pd.to_numeric(df_temp['TotalCharges'].astype(str).str.strip(), errors='coerce').fillna(0.0)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)

    # 1. Target distribution
    sns.countplot(data=df_temp, x='Churn', hue='Churn', palette=['#2b5c8f', '#d95f02'], ax=axes[0, 0], legend=False)
    axes[0, 0].set_title('1. Target Distribution (Churn Imbalance)', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Number of Customers')
    total = len(df_temp)
    for p in axes[0, 0].patches:
        h = p.get_height()
        axes[0, 0].annotate(f'{int(h)} ({h/total*100:.1f}%)', (p.get_x() + p.get_width()/2., h/2),
                            ha='center', va='center', color='white', fontweight='bold', fontsize=11)

    # 2. Churn by Contract
    contract_churn = df_temp.groupby('Contract')['Churn'].value_counts(normalize=True).unstack() * 100
    contract_churn.plot(kind='bar', stacked=True, color=['#2b5c8f', '#d95f02'], ax=axes[0, 1], rot=0)
    axes[0, 1].set_title('2. Churn Rate by Contract Type', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Percentage (%)')
    axes[0, 1].legend(title='Churn', loc='upper right')

    # 3. Churn by Internet Service
    internet_churn = df_temp.groupby('InternetService')['Churn'].value_counts(normalize=True).unstack() * 100
    internet_churn.plot(kind='bar', stacked=True, color=['#2b5c8f', '#d95f02'], ax=axes[0, 2], rot=0)
    axes[0, 2].set_title('3. Churn Rate by Internet Service', fontsize=12, fontweight='bold')
    axes[0, 2].set_ylabel('Percentage (%)')
    axes[0, 2].legend(title='Churn', loc='upper right')

    # 4. Tenure distribution by Churn
    sns.kdeplot(data=df_temp, x='tenure', hue='Churn', palette=['#2b5c8f', '#d95f02'], common_norm=False, fill=True, ax=axes[1, 0])
    axes[1, 0].set_title('4. Tenure Distribution by Churn Status', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Tenure (Months)')

    # 5. Monthly Charges distribution by Churn
    sns.kdeplot(data=df_temp, x='MonthlyCharges', hue='Churn', palette=['#2b5c8f', '#d95f02'], common_norm=False, fill=True, ax=axes[1, 1])
    axes[1, 1].set_title('5. Monthly Charges Distribution by Churn Status', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Monthly Charges ($)')

    # 6. Numerical Correlation Heatmap
    df_corr = df_temp[['tenure', 'MonthlyCharges', 'TotalCharges_Num']].copy()
    df_corr.rename(columns={'TotalCharges_Num': 'TotalCharges'}, inplace=True)
    df_corr['Churn_Binary'] = df_temp['Churn'].map({'Yes': 1, 'No': 0})
    corr = df_corr.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, ax=axes[1, 2], square=True)
    axes[1, 2].set_title('6. Numerical Features Correlation Heatmap', fontsize=12, fontweight='bold')

    plt.suptitle('Exploratory Data Analysis: Key Feature Relationships & Churn Patterns', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    eda_plot_path = os.path.join(plots_dir, 'eda_relationships.png')
    plt.savefig(eda_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[ACTION] Saved exploratory relationship graph: {eda_plot_path}")

def clean_and_prepare(df: pd.DataFrame):
    """
    Clean the dataset based on the EDA findings:
    - Drop customerID
    - Remove 22 duplicate feature records
    - Convert TotalCharges to float and impute 0.0 for tenure=0
    - Encode target Churn ('Yes' -> 1, 'No' -> 0)
    """
    print(f"\n[STEP 3: DATA CLEANING & TRANSFORMATION]")
    df_clean = df.copy()

    # 1. Drop non-predictive identifier
    if 'customerID' in df_clean.columns:
        df_clean = df_clean.drop(columns=['customerID'])
        print("[ACTION] Dropped non-predictive identifier 'customerID'.")

    # 2. Handle duplicates in feature space
    dups_before = df_clean.duplicated().sum()
    if dups_before > 0:
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
        print(f"[ACTION] Detected and removed {dups_before} duplicate customer profiles. Cleaned rows: {len(df_clean)}.")

    # 3. Convert TotalCharges to numeric
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'].astype(str).str.strip(), errors='coerce')
    missing_count = df_clean['TotalCharges'].isnull().sum()
    df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(0.0)
    print(f"[ACTION] Converted 'TotalCharges' to float64; imputed {missing_count} NaN values with 0.0.")

    # 4. Encode target variable
    y = df_clean['Churn'].map({'Yes': 1, 'No': 0}).astype(int)
    X = df_clean.drop(columns=['Churn'])
    print("[ACTION] Encoded target 'Churn': {'Yes': 1, 'No': 0}.")

    return X, y

def get_preprocessor(X: pd.DataFrame):
    """
    Define ColumnTransformer for numerical scaling and categorical encoding.
    Fitted strictly on training data only to prevent data leakage.
    """
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_cols = [c for c in X.columns if c not in numerical_cols]

    print(f"\n[STEP 4: PREPROCESSING PIPELINE DEFINITION]")
    print(f"Numerical Features ({len(numerical_cols)}): {numerical_cols} -> StandardScaler()")
    print(f"Categorical Features ({len(categorical_cols)}): {categorical_cols} -> OneHotEncoder(drop='first')")

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'
    )
    return preprocessor, numerical_cols, categorical_cols

def train_and_evaluate_all():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, 'Dataset', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    model_dir = os.path.join(base_dir, 'model')
    plots_dir = os.path.join(base_dir, 'plots')
    test_data_path = os.path.join(base_dir, 'test_data.csv')

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Load data
    raw_df = load_data(dataset_path)

    # 2. Step-by-step exploratory data analysis & relationship plotting
    explore_dataset(raw_df, plots_dir)

    # 3. Clean and prepare features and target
    X, y = clean_and_prepare(raw_df)

    # 4. Train-Test Split (80:20 Stratified with unique seed)
    print(f"\n[STEP 5: STRATIFIED TRAIN-TEST SPLIT (80:20 with seed={RANDOM_STATE})]")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"Train Set : {X_train.shape[0]} rows | Churn distribution: {np.bincount(y_train)} (Ratio: {np.bincount(y_train)[0]/np.bincount(y_train)[1]:.2f}:1)")
    print(f"Test Set  : {X_test.shape[0]} rows | Churn distribution: {np.bincount(y_test)} (Ratio: {np.bincount(y_test)[0]/np.bincount(y_test)[1]:.2f}:1)")

    # 5. Export test_data.csv to root directory as mandated by rubric
    test_export_df = X_test.copy()
    test_export_df['Churn'] = y_test.map({1: 'Yes', 0: 'No'})
    test_export_df.to_csv(test_data_path, index=False)
    print(f"[ACTION] Exported held-out test split to '{test_data_path}' ({len(test_export_df)} rows).")

    # 6. Fit Preprocessor strictly on Train
    print(f"\n[STEP 6: FIT PREPROCESSING PIPELINE (LEAK-FREE)]")
    preprocessor, num_cols, cat_cols = get_preprocessor(X_train)
    
    print("Fitting ColumnTransformer on X_train only...")
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    encoded_cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
    feature_names = num_cols + list(encoded_cat_names)
    print(f"Transformed feature matrix dimensionality: {X_train_trans.shape[1]} columns")

    # Save fitted preprocessor
    joblib.dump(preprocessor, os.path.join(model_dir, 'preprocessor.pkl'))

    # 7. Define 5 Classification Models
    print(f"\n[STEP 7: MODEL TRAINING & METRICS EVALUATION]")
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver='lbfgs',
            random_state=RANDOM_STATE
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=RANDOM_STATE
        ),
        'kNN': KNeighborsClassifier(
            n_neighbors=7,
            weights='uniform',
            metric='minkowski',
            p=2
        ),
        'Naive Bayes': GaussianNB(),
        'Random Forest (Ensemble)': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=RANDOM_STATE
        )
    }

    results = []
    roc_curves_data = {}
    confusion_matrices_data = {}

    for name, model in models.items():
        print(f"\n---> Training: {name}...")
        model.fit(X_train_trans, y_train)

        # Predictions
        y_pred = model.predict(X_test_trans)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_trans)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test_trans)
        else:
            y_prob = y_pred

        # Compute 6 metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        results.append({
            'ML Model Name': name,
            'Accuracy': acc,
            'AUC': auc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'MCC': mcc
        })

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_curves_data[name] = (fpr, tpr, auc)
        confusion_matrices_data[name] = confusion_matrix(y_test, y_pred)

        # Save model artifact
        safe_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        model_filename = os.path.join(model_dir, f"{safe_name}.pkl")
        joblib.dump(model, model_filename)

        print(f"      Accuracy: {acc:.4f} | AUC: {auc:.4f} | Precision: {prec:.4f}")
        print(f"      Recall  : {rec:.4f} | F1 : {f1:.4f}  | MCC      : {mcc:.4f}")

    # 8. Comparison Table
    metrics_df = pd.DataFrame(results)
    metrics_csv_path = os.path.join(model_dir, 'metrics_comparison.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)

    display_df = metrics_df.copy()
    for col in ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")

    print("\n" + "="*80)
    print("FINAL MODEL COMPARISON TABLE (Test Set N=1,405):")
    print("="*80)
    print(display_df.to_string(index=False))

    # 9. Visualizations
    print(f"\n[STEP 8: GENERATING EVALUATION VISUALIZATION ARTIFACTS]")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Plot 1: Metrics comparison
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    plot_df = pd.melt(metrics_df, id_vars=['ML Model Name'], var_name='Metric', value_name='Score')
    sns.barplot(data=plot_df, x='Metric', y='Score', hue='ML Model Name', palette='tab10', ax=ax)
    ax.set_title('Classification Performance Comparison Across ML Models (Telco Customer Churn)', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Evaluation Metric', fontsize=12, fontweight='bold')
    ax.legend(title='Machine Learning Model', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    
    for p in ax.patches:
        h = p.get_height()
        if not np.isnan(h) and h > 0:
            ax.annotate(f"{h:.2f}",
                        (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom',
                        fontsize=7, color='black',
                        xytext=(0, 2), textcoords='offset points')

    plt.tight_layout()
    bar_plot_path = os.path.join(plots_dir, 'metrics_comparison.png')
    plt.savefig(bar_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[ACTION] Saved: {bar_plot_path}")

    # Plot 2: ROC Curves
    plt.figure(figsize=(9, 7), dpi=300)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for idx, (name, (fpr, tpr, auc_val)) in enumerate(roc_curves_data.items()):
        plt.plot(fpr, tpr, color=colors[idx % len(colors)], lw=2, label=f"{name} (AUC = {auc_val:.4f})")

    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Chance (AUC = 0.50)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12, fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) Curves Comparison', fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    roc_plot_path = os.path.join(plots_dir, 'roc_curves.png')
    plt.savefig(roc_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[ACTION] Saved: {roc_plot_path}")

    # Plot 3: Confusion Matrices
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=300)
    axes = axes.flatten()

    for idx, (name, cm) in enumerate(confusion_matrices_data.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False,
                    xticklabels=['Pred No', 'Pred Yes'], yticklabels=['Actual No', 'Actual Yes'])
        axes[idx].set_title(f"{name}\nAcc: {metrics_df.loc[metrics_df['ML Model Name']==name, 'Accuracy'].values[0]:.3f} | F1: {metrics_df.loc[metrics_df['ML Model Name']==name, 'F1'].values[0]:.3f}",
                            fontsize=11, fontweight='bold')

    if len(confusion_matrices_data) < len(axes):
        fig.delaxes(axes[-1])

    plt.suptitle('Confusion Matrices Comparison on Test Set (N=1,405)', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    cm_plot_path = os.path.join(plots_dir, 'confusion_matrices.png')
    plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[ACTION] Saved: {cm_plot_path}")

    print("\n[SUCCESS] Pipeline execution finished successfully!")
    return metrics_df

if __name__ == '__main__':
    train_and_evaluate_all()
