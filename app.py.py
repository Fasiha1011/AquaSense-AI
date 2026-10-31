from flask import Flask, request, render_template
import joblib
import numpy as np
from pathlib import Path

app = Flask(__name__)
BASE = Path(__file__).resolve().parent

model = joblib.load(BASE / "water_potability_ann.pkl")
imputer = joblib.load(BASE / "water_potability_imputer.pkl")
scaler = joblib.load(BASE / "water_potability_scaler.pkl")
features = joblib.load(BASE / "water_potability_features.pkl")
threshold = joblib.load(BASE / "water_potability_threshold.pkl")

LABELS = {
    "ph": "pH",
    "Hardness": "Hardness",
    "Solids": "Total dissolved solids",
    "Chloramines": "Chloramines",
    "Sulfate": "Sulfate",
    "Conductivity": "Conductivity",
    "Organic_carbon": "Organic carbon",
    "Trihalomethanes": "Trihalomethanes",
    "Turbidity": "Turbidity",
}

@app.route("/", methods=["GET", "POST"])
def home():
    values = {}
    result = None
    error = None

    if request.method == "POST":
        try:
            row = []
            for feature in features:
                raw = request.form.get(feature, "").strip()
                values[feature] = raw
                row.append(np.nan if raw == "" else float(raw))

            X = np.array(row, dtype=float).reshape(1, -1)
            X = imputer.transform(X)
            X = scaler.transform(X)

            probability = float(model.predict_proba(X)[0, 1])
            label = int(probability >= threshold)

            result = {
                "label": label,
                "title": "Class 1: Potable" if label == 1 else "Class 0: Not Potable",
                "probability": round(probability * 100, 2),
            }
        except (ValueError, TypeError):
            error = "Please enter valid numbers, or leave unknown values blank."

    return render_template(
        "index.html",
        fields=features,
        labels=LABELS,
        values=values,
        result=result,
        error=error,
    )

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
