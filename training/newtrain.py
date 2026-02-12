import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
from scikeras.wrappers import KerasRegressor

import joblib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Load dataset
data = pd.read_csv(r"training/floodpred.csv")


# Drop irrelevant columns
X = data.drop(['Flood Risk', 'Flood Prediction', 'Village/City', 'District'], axis=1, errors='ignore')
y = data['Flood Risk']

# One-hot encode categorical features
X = pd.get_dummies(X, columns=['Land Cover', 'Soil Type', 'Is Coastal'])

# Handle missing values
imputer = KNNImputer(n_neighbors=3)
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling for ANN
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define ANN model
def build_ann():
    model = Sequential([
        Input(shape=(X_train_scaled.shape[1],)),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model


ann = KerasRegressor(model=build_ann, epochs=200, batch_size=32, verbose=0,
                     callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
                     validation_split=0.2)
ann.fit(X_train_scaled, y_train)
y_pred_ann = ann.predict(X_test_scaled)


rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)


xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)


w1, w2, w3 = 0.4, 0.3, 0.3
y_pred_hybrid_manual = (w1 * y_pred_ann) + (w2 * y_pred_rf) + (w3 * y_pred_xgb)


stack = StackingRegressor(
    estimators=[('rf', rf), ('xgb', xgb)],
    final_estimator=LinearRegression()
)
stack.fit(X_train, y_train)
y_pred_stack = stack.predict(X_test)


def evaluate_model(name, y_true, y_pred):
    print(f"{name} → R²: {r2_score(y_true, y_pred):.4f}, RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")

evaluate_model("ANN", y_test, y_pred_ann)
evaluate_model("Random Forest", y_test, y_pred_rf)
evaluate_model("XGBoost", y_test, y_pred_xgb)
evaluate_model("Manual Hybrid", y_test, y_pred_hybrid_manual)
evaluate_model("Stacking Hybrid (RF+XGB)", y_test, y_pred_stack)


ann.model_.save("ann_flood_risk.keras")
joblib.dump(rf, "rf_flood_risk.pkl")
joblib.dump(xgb, "xgb_flood_risk.pkl")
joblib.dump(stack, "stacking_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(imputer, "imputer.pkl")

print("\n✅ All models trained and saved successfully!")