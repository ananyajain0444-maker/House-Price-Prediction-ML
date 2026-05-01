import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pandas as pd

from src.data_preprocessing import load_data, clean_data
from src.feature_engineering import feature_engineering
from src.model_training import train_models
from src.evaluation import evaluate_model
from src.prediction import predict_price
from src.utils import create_folders

# ----------------------------
# STEP 1: Create folders
# ----------------------------
create_folders()

# ----------------------------
# STEP 2: Load & clean data
# ----------------------------
df = load_data()
df = clean_data(df)

# ----------------------------
# STEP 3: Feature engineering
# ----------------------------
df = feature_engineering(df)

# ----------------------------
# STEP 4: EDA
# ----------------------------
sns.pairplot(df)
plt.savefig("images/pairplot.png")

plt.figure()
sns.heatmap(df.corr(), annot=True)
plt.savefig("images/heatmap.png")

# ----------------------------
# STEP 5: Prepare data
# ----------------------------
X = df.drop("price", axis=1)
y = df["price"]

# ----------------------------
# STEP 6: Train models
# ----------------------------
lr, rf, X_test, y_test = train_models(X, y)

# ----------------------------
# STEP 7: Predictions
# ----------------------------
lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)

# ----------------------------
# STEP 8: Evaluation
# ----------------------------
evaluate_model(y_test, lr_pred, "Linear Regression")
evaluate_model(y_test, rf_pred, "Random Forest")

# ----------------------------
# STEP 9: Feature Importance (FIXED)
# ----------------------------
feature_names = X.columns
importances = rf.feature_importances_

feat_imp = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x="Importance", y="Feature", data=feat_imp)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("images/feature_importance.png")

# ----------------------------
# STEP 10: Prediction graph
# ----------------------------
plt.figure()
plt.scatter(y_test, rf_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted")
plt.savefig("images/prediction_vs_actual.png")

# ----------------------------
# STEP 11: Save models
# ----------------------------
joblib.dump(lr, "models/linear_model.pkl")
joblib.dump(rf, "models/random_forest_model.pkl")

# ----------------------------
# STEP 12: Sample prediction
# ----------------------------
sample = [[1500, 3, 2, 5, 1000]]  # last value = price_per_sqft
prediction = predict_price(rf, sample)

print("\nSample Prediction:", prediction)

# Save output
with open("outputs/predictions.txt", "w") as f:
    f.write(str(prediction))