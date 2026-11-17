import json
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request
from xgboost import XGBClassifier

BASE = Path(__file__).parent
model = XGBClassifier()
model.load_model(str(BASE / "galaxy_model.json"))
with open(BASE / "model_features.json") as f:
    FEATURES = json.load(f)

# label, hint, example value
# (the two petroR50 examples are illustrative, not taken from the dataset)
FIELDS = {
    "u": ("u magnitude", "brightness in u band", 21.738),
    "g": ("g magnitude", "brightness in g band", 20.266),
    "r": ("r magnitude", "brightness in r band", 19.324),
    "i": ("i magnitude", "brightness in i band", 18.640),
    "z": ("z magnitude", "brightness in z band", 18.238),
    "petroR50_r": ("petroR50_r", "half-light radius, r band (arcsec)", 1.5),
    "petroR50_g": ("petroR50_g", "half-light radius, g band (arcsec)", 1.7),
    "petroRad_r": ("petroRad_r", "Petrosian radius, r band (arcsec)", 3.462),
    "petroRad_g": ("petroRad_g", "Petrosian radius, g band (arcsec)", 4.253),
    "expAB_r": ("expAB_r", "axis ratio of the fitted shape, 0 to 1", 0.289),
}

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    values, result, error = {}, None, None

    if request.method == "POST":
        values = {k: request.form.get(k, "").strip() for k in FEATURES}
        try:
            row = {k: float(v) for k, v in values.items()}
        except ValueError:
            error = "Please fill every box with a number."
        else:
            if not 0 <= row["expAB_r"] <= 1:
                error = "expAB_r must be between 0 and 1."
            elif any(row[k] <= 0 for k in
                     ("petroR50_r", "petroR50_g", "petroRad_r", "petroRad_g")):
                error = "Radii must be positive numbers."
            else:
                X = pd.DataFrame([row], columns=FEATURES)
                prob = float(model.predict_proba(X)[0][1])
                result = {
                    "label": "Starburst" if prob >= 0.5 else "Starforming",
                    "prob_starburst": round(prob * 100, 1),
                }

    return render_template("index.html", fields=FIELDS, features=FEATURES,
                           values=values, result=result, error=error)


if __name__ == "__main__":
    app.run(debug=True)
