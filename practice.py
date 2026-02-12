import joblib
import os
from tensorflow.keras.models import load_model # type: ignore

# Set your model directory
model_dir = "models"

try:
    # Load the scaler and imputer
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    imputer = joblib.load(os.path.join(model_dir, "imputer.pkl"))

    # Load traditional ML models
    rf_model = joblib.load(os.path.join(model_dir, "rf_flood_risk.pkl"))
    xgb_model = joblib.load(os.path.join(model_dir, "xgb_flood_risk.pkl"))
    stack_model = joblib.load(os.path.join(model_dir, "stacking_model.pkl"))

    # Load ANN model (Keras)
    ann_model = load_model(os.path.join(model_dir, "ann_flood_risk.keras"))

    print("✅ TRAINING FEATURES USED BY MODELS:\n")

    print("📐 Scaler trained on:")
    print(scaler.feature_names_in_)

    print("\n🩺 Imputer trained on:")
    print(imputer.feature_names_in_)

    if hasattr(rf_model, "feature_names_in_"):
        print("\n🌲 Random Forest trained on:")
        print(rf_model.feature_names_in_)

    if hasattr(xgb_model, "feature_names_in_"):
        print("\n📦 XGBoost trained on:")
        print(xgb_model.feature_names_in_)

    if hasattr(stack_model, "feature_names_in_"):
        print("\n📚 Stacking model trained on:")
        print(stack_model.feature_names_in_)

    print("\n🧠 ANN model does not directly expose feature names.")
    print("⚠️ Ensure the input shape matches: ", ann_model.input_shape)

except Exception as e:
    print(f"\n❌ Error occurred: {e}")
