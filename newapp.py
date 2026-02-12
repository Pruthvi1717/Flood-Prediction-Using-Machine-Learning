from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load models and pre-processing objects
try:
    model_dir = "models"
    ann_model = load_model(os.path.join(model_dir, "ann_flood_risk.keras"))
    rf_model = joblib.load(os.path.join(model_dir, "rf_flood_risk.pkl"))
    xgb_model = joblib.load(os.path.join(model_dir, "xgb_flood_risk.pkl"))
    stack_model = joblib.load(os.path.join(model_dir, "stacking_model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    imputer = joblib.load(os.path.join(model_dir, "imputer.pkl"))
    print("\n✅ All models and preprocessors loaded successfully")
except Exception as e:
    print(f"\n❌ Failed to load models: {e}")

@app.route("/")
def home():
    return render_template("newpred.html")

@app.route("/newpredict", methods=["GET", "POST"])
def new_predict():
    if request.method == "POST":
        try:
            # 🔹 Step 1: Collect form input
            raw_input = {
                "Latitude": float(request.form.get("latitude", 0)),
                "Longitude": float(request.form.get("longitude", 0)),
                "Rainfall (mm)": float(request.form.get("rainfall", 0)),
                "River Discharge (m³/s)": float(request.form.get("discharge", 0)),
                "Elevation (m)": float(request.form.get("elevation", 0)),
                "Distance From Coast (km)": float(request.form.get("dist_coast", 0)),
                "Population Density": float(request.form.get("population", 0)),
                "Land Cover": request.form.get("land_cover", ""),
                "Soil Type": request.form.get("soil_type", ""),
                "Is Coastal": int(request.form.get("is_coastal", 0)),
            }

            # 🔹 Step 2: One-hot encode categorical fields
            input_data = {  # initialize all with 0
                **{col: 0 for col in scaler.feature_names_in_},
                **{k: raw_input[k] for k in raw_input if k in ["Latitude", "Longitude", "Rainfall (mm)", "River Discharge (m³/s)", "Elevation (m)", "Distance From Coast (km)", "Population Density"]}
            }

            # One-hot encoded columns
            if f"Land Cover_{raw_input['Land Cover']}" in input_data:
                input_data[f"Land Cover_{raw_input['Land Cover']}"] = 1
            if f"Soil Type_{raw_input['Soil Type']}" in input_data:
                input_data[f"Soil Type_{raw_input['Soil Type']}"] = 1
            input_data[f"Is Coastal_{raw_input['Is Coastal']}"] = 1

            # 🔹 Step 3: Convert to DataFrame
            input_df = pd.DataFrame([input_data])
            input_df = input_df[scaler.feature_names_in_]

            # 🔹 Step 4: Impute & scale
            input_imputed = imputer.transform(input_df)
            input_scaled = scaler.transform(input_imputed)

            # 🔹 Step 5: Predict
            pred_ann = ann_model.predict(input_scaled)[0][0] * 100
            pred_rf = rf_model.predict(input_df)[0] * 100
            pred_xgb = xgb_model.predict(input_df)[0] * 100
            pred_stack = stack_model.predict(input_df)[0] * 100
            manual_hybrid = 0.4 * pred_ann + 0.3 * pred_rf + 0.3 * pred_xgb

            results = {
                "ANN Prediction": f"{pred_ann:.2f}%",
                "Random Forest Prediction": f"{pred_rf:.2f}%",
                "XGBoost Prediction": f"{pred_xgb:.2f}%",
                "Manual Hybrid": f"{manual_hybrid:.2f}%",
                "Stacking Model": f"{pred_stack:.2f}%"
            }

            return render_template("newpred.html", prediction=results)

        except Exception as e:
            return f"⚠️ Error in prediction: {e}"

    return render_template("newpred.html")

if __name__ == "__main__":
    app.run(debug=True)
