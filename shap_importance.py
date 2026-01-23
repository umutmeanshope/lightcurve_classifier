import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Load and Prepare Data (Same as your code) ---
model = joblib.load(r"results/model_random_forest.pkl")
rf_model = model.named_steps['model']

X_test = pd.read_pickle("X_test.pkl")
# X_train is not strictly needed for the plot unless used for background data
# X_train = pd.read_pickle("X_train.pkl") 

# Apply transformations
X_temp = model.named_steps['imputer'].transform(X_test)
X_temp = model.named_steps['scaler'].transform(X_temp)
X_test_ready = pd.DataFrame(X_temp, columns=X_test.columns)

# --- 2. Calculate SHAP Values ---
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test_ready)

# --- 3. Handle Dimensions (Your logic, preserved) ---
# We isolate the SHAP values for the Positive Class (index 1)
if isinstance(shap_values, list):
    # Classic Scikit-Learn Random Forest format: [Class0, Class1]
    vals = shap_values[1]
elif len(shap_values.shape) == 3:
    # Newer SHAP format: (Samples, Features, Classes)
    vals = shap_values[:, :, 1]
else:
    # Binary format or already correct
    vals = shap_values

# --- 4. Plotting "Traditional" SHAP Graphs ---

# OPTION A: The Beeswarm Plot (Most detailed and popular)
# This shows importance AND direction (e.g., "High Feature X leads to Low Prediction")

plt.figure()
shap.summary_plot(vals, X_test_ready, show=False)
plt.title("SHAP Summary Plot (Beeswarm)", fontsize=16)
plt.tight_layout()
plt.savefig("results/shap_beeswarm.png", dpi=300, bbox_inches='tight')
plt.show()

# OPTION B: The Standard Bar Plot (Global Importance)
# This is the direct replacement for your manual Seaborn code

plt.figure()
shap.summary_plot(vals, X_test_ready, plot_type="bar", show=False)
plt.title("Mean |SHAP Value| (Global Importance)", fontsize=16)
plt.tight_layout()
plt.savefig("results/shap_standard_bar.png", dpi=300, bbox_inches='tight')
plt.show()

print("Traditional SHAP plots generated successfully.")