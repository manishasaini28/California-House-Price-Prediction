from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load Trained Model

with open("model.pkl", "rb") as file:
    model = pickle.load(file)


# Home Page

@app.route("/")
def home():
    return render_template("index.html")


# Dashboard Page

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


# Prediction

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = {
            "longitude": [float(request.form["longitude"])],
            "latitude": [float(request.form["latitude"])],
            "housing_median_age": [float(request.form["housing_median_age"])],
            "total_rooms": [float(request.form["total_rooms"])],
            "total_bedrooms": [float(request.form["total_bedrooms"])],
            "population": [float(request.form["population"])],
            "households": [float(request.form["households"])],
            "median_income": [float(request.form["median_income"])],
            "ocean_proximity": [request.form["ocean_proximity"]]
        }

        df = pd.DataFrame(data)

        prediction = model.predict(df)[0]

        prediction = f"${prediction:,.2f}"

        return render_template(
            "index.html",
            prediction=prediction
        )

    except Exception as e:

        return render_template(
            "index.html",
            prediction=f"Error : {str(e)}"
        )


# Run App

if __name__ == "__main__":
    app.run(debug=True , port=5002)