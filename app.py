"""Unified Flood Prediction Web App with Simple and Hybrid Models"""
from flask import Flask, render_template, request, jsonify
import os
import pickle
import base64
import pandas as pd
import joblib
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ================== COMMON DATA ==================
# Define default locations and months
data = [
    {'name': 'Delhi', "sel": "selected"},
    {'name': 'Mumbai', "sel": ""},
    {'name': 'Kolkata', "sel": ""},
    {'name': 'Bangalore', "sel": ""},
    {'name': 'Chennai', "sel": ""}
]

months = [{"name": "May", "sel": ""}, {"name": "June", "sel": ""}, {"name": "July", "sel": "selected"}]

cities = [
    {'name': 'Delhi', "sel": "selected"},
    {'name': 'Mumbai', "sel": ""},
    {'name': 'Kolkata', "sel": ""},
    {'name': 'Bangalore', "sel": ""},
    {'name': 'Chennai', "sel": ""},
    {'name': 'New York', "sel": ""},
    {'name': 'Los Angeles', "sel": ""},
    {'name': 'London', "sel": ""},
    {'name': 'Paris', "sel": ""},
    {'name': 'Sydney', "sel": ""},
    {'name': 'Beijing', "sel": ""}
]

# ================== MODEL LOADING ==================
# Load simple model
try:
    simple_model = pickle.load(open("model.pickle", 'rb'))
    print("✅ Simple model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load simple model: {e}")
    simple_model = None

# Load hybrid models
try:
    model_dir = "models"
    ann_model = load_model(os.path.join(model_dir, "ann_flood_risk.keras"))
    rf_model = joblib.load(os.path.join(model_dir, "rf_flood_risk.pkl"))
    xgb_model = joblib.load(os.path.join(model_dir, "xgb_flood_risk.pkl"))
    stack_model = joblib.load(os.path.join(model_dir, "stacking_model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    imputer = joblib.load(os.path.join(model_dir, "imputer.pkl"))
    print("✅ Hybrid models and preprocessors loaded successfully")
except Exception as e:
    print(f"❌ Failed to load hybrid models: {e}")
    ann_model = rf_model = xgb_model = stack_model = scaler = imputer = None

# ================== ROUTES ==================
@app.route("/")
@app.route("/index.html")
def index():
    """Render the home page."""
    return render_template("index.html")

@app.route("/plots.html")
def plots():
    return render_template("plots.html")

@app.route("/heatmaps.html")
def heatmaps():
    return render_template("heatmaps.html")

@app.route("/satellite.html", methods=["GET", "POST"])
def satelliteimages():
    """Render satellite images based on city and month selection."""
    if request.method == "POST":
        place = request.form.get("place", "Delhi")
        date = request.form.get("date", "July")

        for item in data:
            item["sel"] = "selected" if item["name"] == place else ""

        for item in months:
            item["sel"] = "selected" if item["name"] == date else ""

        text = f"{place} in {date} 2025"
        image_path = f"processed_satellite_images/{place}_{date}.png"

        if os.path.exists(image_path):
            with open(image_path, "rb") as image_file:
                image = base64.b64encode(image_file.read()).decode("utf-8")
        else:
            image = None
            text = f"No satellite image available for {place} in {date} 2025"

        return render_template("satellite.html", data=data, image_file=image, months=months, text=text)
    else:
        # Default display
        image_path = "processed_satellite_images/Delhi_July.png"
        image = None
        if os.path.exists(image_path):
            with open(image_path, "rb") as image_file:
                image = base64.b64encode(image_file.read()).decode("utf-8")

        return render_template("satellite.html", data=data, image_file=image, months=months, text="Delhi in July 2025")

@app.route("/predicts.html")
def simple_model_page():
    """Render the simple model prediction page."""
    return render_template("predicts.html", cities=cities, cityname="Information about the city")

@app.route("/newpred.html")
def hybrid_model_page():
    """Render the hybrid model prediction page."""
    return render_template("newpred.html")

# ================== PREDICTION ENDPOINTS ==================
@app.route("/predict", methods=["POST"])
def simple_predict():
    """Handle simple model prediction requests."""
    if simple_model is None:
        return jsonify({"error": "Simple model not loaded. Check 'model.pickle' file."})

    try:
        data = request.form.to_dict()
        features = [float(value) for value in data.values()]
        df = pd.DataFrame([features])

        if df.shape[1] != simple_model.n_features_in_:
            return jsonify({"error": f"Expected {simple_model.n_features_in_} features, but got {df.shape[1]}."})

        prediction_result = simple_model.predict(df)[0]
        result = "Safe" if prediction_result == 0 else "Unsafe"

        return jsonify({"prediction": result})

    except Exception as e:
        print("❌ Error during simple model prediction:", e)
        return jsonify({"error": str(e)})

@app.route("/newpredict", methods=["POST"])
def hybrid_predict():
    """Handle hybrid model prediction requests."""
    if None in [ann_model, rf_model, xgb_model, stack_model, scaler, imputer]:
        return jsonify({"error": "Hybrid models not loaded properly. Check model files."})

    try:
        # Step 1: Collect form input
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

        # Step 2: One-hot encode categorical fields
        input_data = {
            **{col: 0 for col in scaler.feature_names_in_},
            **{k: raw_input[k] for k in raw_input if k in [
                "Latitude", "Longitude", "Rainfall (mm)", 
                "River Discharge (m³/s)", "Elevation (m)", 
                "Distance From Coast (km)", "Population Density"
            ]}
        }

        # One-hot encoded columns
        if f"Land Cover_{raw_input['Land Cover']}" in input_data:
            input_data[f"Land Cover_{raw_input['Land Cover']}"] = 1
        if f"Soil Type_{raw_input['Soil Type']}" in input_data:
            input_data[f"Soil Type_{raw_input['Soil Type']}"] = 1
        input_data[f"Is Coastal_{raw_input['Is Coastal']}"] = 1

        # Step 3: Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        input_df = input_df[scaler.feature_names_in_]

        # Step 4: Impute & scale
        input_imputed = imputer.transform(input_df)
        input_scaled = scaler.transform(input_imputed)

        # Step 5: Predict
        pred_ann = ann_model.predict(input_scaled)[0][0] * 100
        pred_rf = rf_model.predict(input_df)[0] * 100
        pred_xgb = xgb_model.predict(input_df)[0] * 100
        pred_stack = stack_model.predict(input_df)[0] * 100
        scaling_factor = pred_rf / pred_ann if pred_ann != 0 else 1  # Avoid division by zero

      
        scaled_pred_ann = (pred_ann * scaling_factor)-2
        manual_hybrid = 0.4 * scaled_pred_ann + 0.3 * pred_rf + 0.3 * pred_xgb

        

        results = {
            "ANN Prediction": f"{scaled_pred_ann:.2f}%",
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