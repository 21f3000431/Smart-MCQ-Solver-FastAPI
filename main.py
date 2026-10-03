import joblib
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent

vectorizer = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")
model = joblib.load(BASE_DIR / "logistic_model.pkl")

app = FastAPI(title="Smart MCQ Solver API")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True
    }


class MCQRequest(BaseModel):
    question: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    option_e: str


@app.post("/predict")
def predict(data: MCQRequest):

    combined_text = (
        f"{data.question} "
        f"A: {data.option_a} "
        f"B: {data.option_b} "
        f"C: {data.option_c} "
        f"D: {data.option_d} "
        f"E: {data.option_e}"
    )

    vectorized_input = vectorizer.transform([combined_text])

    probabilities = model.predict_proba(vectorized_input)[0]

    top3_idx = probabilities.argsort()[-3:][::-1]

    return {
        "predicted_class": model.classes_[top3_idx[0]],
        "top_3_predictions": [
            {
                "class": model.classes_[i],
                "probability": probabilities[i]
            }
            for i in top3_idx
        ]
    }