from fastapi import FastAPI
import joblib

app = FastAPI()

model = joblib.load("models/random_forest_model.pkl")

@app.get("/")
def home():
    return {"message": "House Price Prediction API Running"}

@app.post("/predict")
def predict(data: dict):
    features = [[
        data["area"],
        data["bedrooms"],
        data["bathrooms"],
        data["age"]
    ]]

    prediction = model.predict(features)[0]
    return {"predicted_price": prediction}