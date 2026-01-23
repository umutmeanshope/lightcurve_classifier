import joblib
import numpy as np
from sklearn.ensemble import VotingClassifier

from split import split_data
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SHAPE_FEATURES = True  # TOGGLE TRUE TO USE SHAPE FEATURES

RF_TRESH = 0.48
SVM_TRESH = 0.40

rf_model = joblib.load("results/model_random_forest.pkl")
svm_model = joblib.load("results/model_svm.pkl")

X_train, X_test, y_train, y_test = split_data()

# ==========================================
# Cascade Classifier Experiment
# Stage 1: SVM (The Net) -> Stage 2: RF (The Filter)
# ==========================================

# --- STAGE 1: SVM ---
# We use a loose threshold (0.40) to make sure we catch everything
stage1_threshold = SVM_TRESH
svm_probs = svm_model.predict_proba(X_test)[:, 1]

# These are the candidates SVM thinks "might" be planets
stage1_candidates_mask = svm_probs >= stage1_threshold

# --- STAGE 2: Random Forest ---
# We ONLY look at the stars that passed Stage 1
# We use the RF's strict threshold (0.48)
rf_probs = rf_model.predict_proba(X_test)[:, 1]

# Apply the Cascade Logic
# A star is a flagged ONLY IF:
# (SVM says Yes) AND (Random Forest says Yes)
final_cascade_preds = (
    (svm_probs >= SVM_TRESH) &     # Passed the Net
    (rf_probs >= RF_TRESH)   # Passed the Inspection
).astype(int)

# --- Evaluate ---
from sklearn.metrics import precision_score, recall_score, f1_score

c_prec = precision_score(y_test, final_cascade_preds)
c_rec = recall_score(y_test, final_cascade_preds)
c_f1 = f1_score(y_test, final_cascade_preds)

print(f"--- Cascade Pipeline Results ---")
print(f"Precision: {round(c_prec, 3)}")
print(f"Recall:    {round(c_rec, 3)}")
print(f"F1 Score:  {round(c_f1, 3)}")

# --- SAVE ---
results_df = pd.DataFrame({
    'Model': ['Cascade Classifier (SVM+RF)'],
    'Best Threshold': [f"-"],
    'Precision': [round(c_prec, 3)],
    'Recall': [round(c_rec, 3)],
    'F1 Score': [round(c_f1, 3)]
})
results_df.to_pickle("results/cascade_metrics_report.pkl")
print("Saved metrics to 'cascade_metrics_report.pkl'")

# --- PLOT & SAVE CONFUSION MATRIX ---
cm = confusion_matrix(y_test, final_cascade_preds)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Not Planet', 'Planet'],
            yticklabels=['Not Planet', 'Planet'])

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Cascade Classifier Confusion Matrix\n(SVM > 0.40 & RF > 0.48)')

# Save the plot
plt.savefig(f"results/cascade_confusion_matrix.png", dpi=300, bbox_inches='tight')
print("Saved plot to 'cascade_confusion_matrix.png'")
plt.show()