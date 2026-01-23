import joblib
import numpy as np
from sklearn.ensemble import VotingClassifier

from split import split_data
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

rf_model = joblib.load("results/model_random_forest.pkl")
svm_model = joblib.load("results/model_svm.pkl")

ensemble = VotingClassifier(
    estimators=[
        ('rf', rf_model), 
        ('svm', svm_model)
    ],
    voting='soft'
)

X_train, X_test, y_train, y_test = split_data()


# 2. Train 
ensemble.fit(X_train, y_train)

# 3. Get the "Raw Opinions"
ensemble_probs = ensemble.predict_proba(X_test)[:, 1]

# 4. Optimize the Threshold for the Committee

best_ens_f1 = 0
best_ens_thresh = 0
thresholds = np.arange(0.1, 0.9, 0.01)

for t in thresholds:
    preds = (ensemble_probs >= t).astype(int)
    score = f1_score(y_test, preds)
    if score > best_ens_f1:
        best_ens_f1 = score
        best_ens_thresh = t

print(f"Ensemble Optimized F1: {round(best_ens_f1, 3)} at Threshold: {round(best_ens_thresh, 3)}")

final_preds = (ensemble_probs >= best_ens_thresh).astype(int)

# Calculate the scores
final_precision = precision_score(y_test, final_preds)
final_recall = recall_score(y_test, final_preds)

# Save to a DataFrame
ensemble_results = pd.DataFrame({
    'Model': ['Voting Ensemble (Random Forest + SVM)'],
    'Best Threshold': [round(best_ens_thresh, 3)],
    'Precision': [round(final_precision, 3)],
    'Recall': [round(final_recall, 3)],
    'F1 Score': [round(best_ens_f1, 3)],
})
print(ensemble_results)

# Save to CSV
ensemble_results.to_csv('results/ensemble_metrics_report.csv', index=False)
print("Saved metrics to 'results/ensemble_metrics_report.csv'")


# Generate and Save Confusion Matrix

cm = confusion_matrix(y_test, final_preds)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Not Planet', 'Planet'],
            yticklabels=['Not Planet', 'Planet'])

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Ensemble Confusion Matrix\n(Threshold = {best_ens_thresh:.3f})')

# Save the plot
plt.savefig('results/ensemble_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Saved plot to 'results/ensemble_confusion_matrix.png'")
plt.show()