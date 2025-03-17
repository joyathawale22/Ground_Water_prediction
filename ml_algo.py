import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

print("Loading combined data...")
# -------------------------------
# 1. Load Data
# -------------------------------
file_path = r"D:\Ground water quality major project\combined_data.csv"
data = pd.read_csv(file_path, low_memory=False)
print("Data loaded. Shape:", data.shape)

# -------------------------------
# 2. Feature Engineering
# -------------------------------
print("Performing feature engineering...")
# Calculate average pH and round it to two decimals for consistency.
data['Avg pH Raw'] = ((data['pH Min'] + data['pH Max']) / 2).round(2)
data['Avg Temperature'] = (data['Temperature Min'] + data['Temperature Max']) / 2
data['Avg Conductivity'] = (data['Conductivity (µmhos/cm) Min'] + data['Conductivity (µmhos/cm) Max']) / 2

# Use a fixed threshold of pH=7.0 for Safe vs. Unsafe
data['Water Quality'] = data['Avg pH Raw'].apply(lambda x: 1 if x >= 7.00 else 0)
print("Feature engineering complete.")

# -------------------------------
# 3. Define Features and Split Data
# -------------------------------
print("Splitting data into train and test sets...")
features = [
    'Temperature Min', 'Temperature Max',
    'Conductivity (µmhos/cm) Min', 'Conductivity (µmhos/cm) Max',
    'Year', 'Avg Temperature', 'Avg Conductivity'
]

X_reg = data[features]
y_reg = data['Avg pH Raw']

X_clf = data[features]
y_clf = data['Water Quality']

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)
X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)
print("Data splitting complete.")

# -------------------------------
# 4. Scaling Features
# -------------------------------
print("Scaling features...")
scaler = StandardScaler()
X_reg_train_scaled = scaler.fit_transform(X_reg_train)
X_reg_test_scaled = scaler.transform(X_reg_test)
X_clf_train_scaled = scaler.fit_transform(X_clf_train)
X_clf_test_scaled = scaler.transform(X_clf_test)
print("Scaling complete.")

# -------------------------------
# 5. Train Models
# -------------------------------
print("Starting model training...")

# Regression Models
print("\nTraining Linear Regression model...")
lin_reg = LinearRegression()
lin_reg.fit(X_reg_train_scaled, y_reg_train)
pred_lin = lin_reg.predict(X_reg_test_scaled)
mse_lin = mean_squared_error(y_reg_test, pred_lin)
print("Linear Regression trained.")

print("\nTraining Random Forest Regressor (with verbose)...")
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, verbose=1)
rf_reg.fit(X_reg_train, y_reg_train)
pred_rf = rf_reg.predict(X_reg_test)
mse_rf = mean_squared_error(y_reg_test, pred_rf)
print("Random Forest Regressor trained.")

print("\nTraining Gradient Boosting Regressor (with verbose)...")
gb_reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbose=1)
gb_reg.fit(X_reg_train, y_reg_train)
pred_gb = gb_reg.predict(X_reg_test)
mse_gb = mean_squared_error(y_reg_test, pred_gb)
print("Gradient Boosting Regressor trained.")

# Classification Models
print("\nTraining Logistic Regression (with verbose)...")
log_reg = LogisticRegression(max_iter=1000, random_state=42, verbose=1)
log_reg.fit(X_clf_train_scaled, y_clf_train)
y_pred_log = log_reg.predict(X_clf_test_scaled)
print("Logistic Regression trained.")

print("\nTraining Random Forest Classifier (with verbose)...")
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, verbose=1)
rf_clf.fit(X_clf_train, y_clf_train)
y_pred_rf_clf = rf_clf.predict(X_clf_test)
print("Random Forest Classifier trained.")

print("\nTraining SVM Classifier (with verbose)...")
svm_clf = SVC(kernel='rbf', random_state=42, verbose=True)
svm_clf.fit(X_clf_train_scaled, y_clf_train)
y_pred_svm = svm_clf.predict(X_clf_test_scaled)
print("SVM Classifier trained.")

# -------------------------------
# 6. Create Interactive Dashboard
# -------------------------------
print("\nCreating interactive dashboard...")
fig = make_subplots(
    rows=3, cols=3,
    subplot_titles=(
        "pH Distribution",
        "Water Quality Distribution",
        "Linear Regression (Actual vs. Pred)",
        "Random Forest Regressor (Actual vs. Pred)",
        "Gradient Boosting Regressor (Actual vs. Pred)",
        "Logistic Regression CM",
        "Random Forest Classifier CM",
        "SVM Classifier CM",
        "Model Performance Summary"
    ),
    horizontal_spacing=0.15,
    vertical_spacing=0.1
)

# 1. pH Distribution (Histogram)
fig.add_trace(
    go.Histogram(
        x=data['Avg pH Raw'],
        nbinsx=30,
        name="pH Dist"
    ),
    row=1, col=1
)
fig.add_annotation(
    xref="x1", yref="y1",
    x=0.5, y=-0.2,
    text=f"Min pH: {data['Avg pH Raw'].min():.2f}, Max pH: {data['Avg pH Raw'].max():.2f}",
    showarrow=False,
    font=dict(size=10),
    row=1, col=1
)

# 2. Water Quality Distribution (Bar)
quality_counts = data['Water Quality'].value_counts()
fig.add_trace(
    go.Bar(
        x=quality_counts.index,
        y=quality_counts.values,
        name="Quality Dist"
    ),
    row=1, col=2
)
fig.add_annotation(
    xref="x2", yref="y2",
    x=0.5, y=-0.2,
    text="0 = Unsafe, 1 = Safe",
    showarrow=False,
    font=dict(size=10),
    row=1, col=2
)

# 3. Linear Regression: Actual vs Predicted
fig.add_trace(
    go.Scatter(
        x=y_reg_test,
        y=pred_lin,
        mode='markers',
        name="LinReg Points"
    ),
    row=1, col=3
)
fig.add_trace(
    go.Scatter(
        x=y_reg_test,
        y=y_reg_test,
        mode='lines',
        name="Ideal",
        showlegend=False
    ),
    row=1, col=3
)

# 4. Random Forest Regressor: Actual vs Predicted
fig.add_trace(
    go.Scatter(
        x=y_reg_test,
        y=pred_rf,
        mode='markers',
        name="RFReg Points"
    ),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(
        x=y_reg_test,
        y=y_reg_test,
        mode='lines',
        name="Ideal",
        showlegend=False
    ),
    row=2, col=1
)

# 5. Gradient Boosting Regressor: Actual vs Predicted
fig.add_trace(
    go.Scatter(
        x=y_reg_test,
        y=pred_gb,
        mode='markers',
        name="GBReg Points"
    ),
    row=2, col=2
)
fig.add_trace(
    go.Scatter(
        x=y_reg_test,
        y=y_reg_test,
        mode='lines',
        name="Ideal",
        showlegend=False
    ),
    row=2, col=2
)

# 6. Logistic Regression Confusion Matrix (Heatmap)
cm_log = confusion_matrix(y_clf_test, y_pred_log)
fig.add_trace(
    go.Heatmap(
        z=cm_log,
        x=[0, 1],
        y=[0, 1],
        colorscale='Blues',
        name="LogReg CM",
        showscale=True,
        showlegend=False
    ),
    row=2, col=3
)
fig.add_annotation(
    xref="x6", yref="y6",
    x=0.5, y=-0.2,
    text="0 = Unsafe, 1 = Safe",
    showarrow=False,
    font=dict(size=10),
    row=2, col=3
)

# 7. Random Forest Classifier Confusion Matrix
cm_rf = confusion_matrix(y_clf_test, y_pred_rf_clf)
fig.add_trace(
    go.Heatmap(
        z=cm_rf,
        x=[0, 1],
        y=[0, 1],
        colorscale='Greens',
        name="RFClf CM",
        showscale=True,
        showlegend=False
    ),
    row=3, col=1
)
fig.add_annotation(
    xref="x7", yref="y7",
    x=0.5, y=-0.2,
    text="0 = Unsafe, 1 = Safe",
    showarrow=False,
    font=dict(size=10),
    row=3, col=1
)

# 8. SVM Classifier Confusion Matrix
cm_svm = confusion_matrix(y_clf_test, y_pred_svm)
fig.add_trace(
    go.Heatmap(
        z=cm_svm,
        x=[0, 1],
        y=[0, 1],
        colorscale='Reds',
        name="SVMClf CM",
        showscale=True,
        showlegend=False
    ),
    row=3, col=2
)
fig.add_annotation(
    xref="x8", yref="y8",
    x=0.5, y=-0.2,
    text="0 = Unsafe, 1 = Safe",
    showarrow=False,
    font=dict(size=10),
    row=3, col=2
)

# 9. Model Performance Summary
summary_stats = {
    "LinReg MSE": mse_lin,
    "RFReg MSE": mse_rf,
    "GBReg MSE": mse_gb,
    "LogReg Acc": accuracy_score(y_clf_test, y_pred_log),
    "RFClf Acc": accuracy_score(y_clf_test, y_pred_rf_clf),
    "SVMClf Acc": accuracy_score(y_clf_test, y_pred_svm)
}
summary_text = "<br>".join([f"{key}: {value:.4f}" for key, value in summary_stats.items()])

safe_count = np.sum(y_pred_log == 1)
unsafe_count = np.sum(y_pred_log == 0)
suitability_text = f"Water Quality Suitability:<br>Safe: {safe_count} samples<br>Unsafe: {unsafe_count} samples"
summary_text += f"<br><br>{suitability_text}"

fig.add_trace(
    go.Scatter(
        x=[0],
        y=[0],
        mode="text",
        text=[summary_text],
        textposition="middle center",
        showlegend=False
    ),
    row=3, col=3
)

fig.update_layout(
    title="Groundwater Quality Analysis Dashboard",
    height=1200,
    width=1600,
    template="plotly_white",
    legend={
        "title": "Legend",
        "x": 1.05,
        "y": 1,
        "xanchor": "left",
        "yanchor": "top"
    }
)

fig.show()

# Save the trained models
joblib.dump(rf_reg, "random_forest_regressor.pkl")
joblib.dump(rf_clf, "random_forest_classifier.pkl")

# -------------------------------
# 7. Predict Quality Function with Debug Prints
# -------------------------------
def predict_quality(features):
    """
    Predicts groundwater quality based on input features.
    
    Parameters:
        features (list): A list of numerical features.
            Expected order:
                - If length is 5:
                  [Temperature Min, Temperature Max, Conductivity (µmhos/cm) Min, Conductivity (µmhos/cm) Max, Year]
                  (Avg Temperature and Avg Conductivity will be computed.)
                - If length is 7:
                  [Temperature Min, Temperature Max, Conductivity (µmhos/cm) Min, Conductivity (µmhos/cm) Max, Year, Avg Temperature, Avg Conductivity]
    
    Returns:
        dict: A dictionary with keys 'predicted_pH' and 'water_quality'.
    """
    print("predict_quality() received features:", features)
    
    # If only 5 features are provided, compute the averages
    if len(features) == 5:
        temperature_min, temperature_max, conductivity_min, conductivity_max, year = features
        avg_temperature = (temperature_min + temperature_max) / 2
        avg_conductivity = (conductivity_min + conductivity_max) / 2
        features = [temperature_min, temperature_max, conductivity_min, conductivity_max, year, avg_temperature, avg_conductivity]
        print("Computed averages. New features:", features)
    else:
        print("Using provided 7 features:", features)
    
    # Load the saved models
    rf_reg = joblib.load("random_forest_regressor.pkl")
    rf_clf = joblib.load("random_forest_classifier.pkl")
    
    print("Final features used for prediction:", features)
    
    # Make predictions using the models
    predicted_ph = rf_reg.predict([features])[0]
    predicted_quality = rf_clf.predict([features])[0]
    water_quality = "Safe" if predicted_quality == 1 else "Unsafe"
    
    result = {"predicted_pH": round(predicted_ph, 2), "water_quality": water_quality}
    print("Prediction result:", result)
    return result

# Example usage:
if __name__ == "__main__":
    sample_features = [20, 30, 150, 200, 2023]
    print("Test output:", predict_quality(sample_features))
