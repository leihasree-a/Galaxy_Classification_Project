from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load saved model
with open(r"C:\Users\Admin\OneDrive\Documents\galaxy-classification-project\app\RF.pkl", 'rb') as file:
    model = pickle.load(file)

# Home page
@app.route("/")
def home():
    return render_template("home.html")

# Input page route  (you did NOT add this earlier)
@app.route("/input")
def input_page():
    return render_template("input.html")

@app.route("/input.html")
def input_page_html():
    return render_template("input.html")


# Submit and predict
@app.route("/submit", methods=["POST"])
def submit():
    # Read values from form
    input_features = [float(x) for x in request.form.values()]

    # Your 10 feature names (should match model)
    names = [
        'specobjid',
        'modelFlux_i',
        'modelFlux_z',
        'petroRad_u',
        'petroRad_g',
        'petroRad_i',
        'petroRad_r',
        'petroRad_z',
        'petroR50_z',
        'redshift'
    ]

    # Convert to DataFrame
    data = pd.DataFrame([input_features], columns=names)

    # Predict
    prediction = model.predict(data)[0]

    # Map output
    result = "starforming" if prediction == 0 else "starbursting"

    return render_template("output.html", prediction=result)

# Main
if __name__ == "__main__":
    print("Templates folder exists?", os.path.isdir("templates"))

    app.run(debug=True, port=2222)
