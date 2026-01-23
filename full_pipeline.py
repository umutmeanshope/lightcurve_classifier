from split import split_data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # For saving models
import os
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from imblearn.over_sampling import SMOTE
from scipy.stats import randint, loguniform, uniform
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, classification_report, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score


# --- 1. CONFIGURATION ---

output_folder_name = "results"
# Create a directory to save results

os.makedirs(output_folder_name, exist_ok=True)


print(f"Results will be saved to: {output_folder_name}")

# Y for yes n for no
confirm = input("Proceed with these settings? (y/n): ")
if confirm.lower() != 'y':
    print("Pipeline execution terminated by user.")
    exit()

models_config = [
    {
        "name": "Random Forest",
        "estimator": RandomForestClassifier(random_state=42, n_jobs=1),
        "params": {
            "model__n_estimators": randint(100, 500),
            "model__max_depth": randint(10, 30),
            "model__min_samples_split": randint(2, 21),
            "model__min_samples_leaf": randint(1, 21),
            "model__bootstrap": [True, False],
            "model__criterion": ["gini", "entropy"],
            "model__max_features": ["sqrt", "log2", None]
        }
    },
    {
        "name": "SVM",
        "estimator": SVC(random_state=42, cache_size=1000, probability=True),
        "params": {
            "model__C": loguniform(1e-1, 1e3),
            "model__kernel": ["rbf", "poly", "sigmoid"],
            "model__gamma": ["scale", "auto"],
        }
    },
    {
        "name": "Gradient Boosting",
        "estimator": GradientBoostingClassifier(random_state=42),
        "params": {
            "model__n_estimators": randint(100, 500),
            "model__max_depth": randint(3, 8),
            "model__learning_rate": loguniform(0.01, 0.2),
            "model__subsample": uniform(0.7, 0.3),
            "model__min_samples_split": randint(2, 20),
            "model__min_samples_leaf": randint(1, 20),
            "model__max_features": ["sqrt", "log2"]
        }
    },
    {
        "name": "Logistic Regression",
        "estimator": LogisticRegression(solver='liblinear', random_state=42, max_iter=10000),
        "params": {
            "model__C": loguniform(1e-4, 1e2),
            "model__penalty": ["l1", "l2"],
        }
    },
    {
        "name": "KNN",
        "estimator": KNeighborsClassifier(n_jobs=1),
        "params": {
            "model__n_neighbors": randint(3, 30),
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        }
    },
    {
        "name": "Decision Tree",
        "estimator": DecisionTreeClassifier(random_state=42),
        "params": {
            "model__max_depth": randint(3, 20),
            "model__min_samples_split": randint(2, 21),
            "model__min_samples_leaf": randint(1, 21),
            "model__criterion": ["gini", "entropy"],
            "model__max_features": ["sqrt", "log2", None]
        }
    },
]

# --- 2. TUNING FUNCTION ---
def fine_tuning(model_name, estimator, param_distro, X, y):
    print(f"\n--- Tuning {model_name} ---")
    
    pipeline = ImbPipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('smote', SMOTE(random_state=42)),
        ('scaler', StandardScaler()),
        ('model', estimator)
    ])

    rnd_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distro,
        n_iter=30,  # Try 30 combinations
        cv=5,
        verbose=1,
        random_state=42,
        n_jobs=1,  # sad pc :(
        scoring="f1"
    )

    rnd_search.fit(X, y)
    print(f"Best F1 (CV): {rnd_search.best_score_:.4f}")
    return rnd_search

# --- 3. MAIN EXECUTION LOOP ---
X_train, X_test, y_train, y_test = split_data()

results_list = []
trained_models = {}

# Prepare ROC Plot Figure
plt.figure(figsize=(10, 8))

for config in models_config:
    name = config["name"]
    safe_name = name.replace(' ', '_').lower()
    
    # A. Run Fine Tuning
    search_result = fine_tuning(name, config["estimator"], config["params"], X_train, y_train)
    best_model = search_result.best_estimator_
    trained_models[name] = best_model
    
    # B. Save Model to File
    model_filename = f"{output_folder_name}/model_{safe_name}.pkl"
    joblib.dump(best_model, model_filename)
    
    # C. Get Probabilities for Test Set
    try:
        y_probs = best_model.predict_proba(X_test)[:, 1]
    except AttributeError:
        # Fallback for models that might not have predict_proba enabled by default
        y_probs = best_model.decision_function(X_test)
        y_probs = (y_probs - y_probs.min()) / (y_probs.max() - y_probs.min())

    # D. Determine Best Threshold
    
    
    
    best_f1 = 0.0
    # Search for best threshold
    thresholds = np.arange(0.1, 0.9, 0.01)
    for t in thresholds:
        y_pred_t = (y_probs >= t).astype(int)
        score = f1_score(y_test, y_pred_t)
        if score > best_f1:
            best_f1 = score
            best_thresh = t
    

    # E. Calculate Final Metrics using the chosen threshold
    y_pred_final = (y_probs >= best_thresh).astype(int)
    
    final_f1 = f1_score(y_test, y_pred_final)
    final_precision = precision_score(y_test, y_pred_final)
    final_recall = recall_score(y_test, y_pred_final)
    
    # F. Store Results
    # round to 3 decimal places
    results_list.append({
        "Model": name,
        "Best CV Score (F1)": round(search_result.best_score_, 3),
        "Best Threshold": round(best_thresh, 3),
        "Precision": round(final_precision, 3),
        "Recall": round(final_recall, 3),
        "New F1": round(final_f1, 3),
        "Best Params": str(search_result.best_params_)
    })
    
    # G. Add to ROC Plot (Main Comparison Chart)
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
    
    # --- SAVE CONFUSION MATRIX (Uses the selected threshold) ---
    cm = confusion_matrix(y_test, y_pred_final)
    
    plt.figure(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["False Pos", "Planet"])
    disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
    plt.title(f"Confusion Matrix: {name}\n(Threshold: {best_thresh:.2f})")
    plt.tight_layout()
    plt.savefig(f"{output_folder_name}/confusion_matrix_{safe_name}.png")
    plt.close() 
    
    

# --- 4. FINALIZE OUTPUTS ---

print("Saving final comparison charts...")
plt.figure(figsize=(10, 8))
for name, model in trained_models.items():
    try:
        y_probs = model.predict_proba(X_test)[:, 1]
    except:
        y_probs = model.decision_function(X_test)
        y_probs = (y_probs - y_probs.min()) / (y_probs.max() - y_probs.min())
        
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Final Model Comparison: ROC Curves')
plt.legend(loc="lower right")
plt.savefig(f"{output_folder_name}/roc_curves.png")
plt.close()

# Save Dataframe Report
results_df = pd.DataFrame(results_list)
# Sort by F1
results_df = results_df.sort_values(by="New F1", ascending=False)
results_df.to_csv(f"{output_folder_name}/final_performance_report.csv", index=False)

# Display Final Report
print("\n" + "="*80)
print("FINAL PROJECT REPORT")
print("="*80)
# Show the new columns in the printout
print(results_df[['Model', 'New F1', 'Precision', 'Recall', 'Best Threshold']].to_string(index=False))
print(f"\nAll files saved to: {output_folder_name}/")