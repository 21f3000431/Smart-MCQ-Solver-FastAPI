# Smart MCQ Solver - FastAPI

FastAPI deployment of a Smart MCQ Solver using TF-IDF vectorization and Logistic Regression.

## Features

* Accepts an MCQ question and five answer options.
* Predicts the most likely answer choice (`A`, `B`, `C`, `D`, or `E`).
* Returns the top 3 predictions with their probabilities.
* Provides a health-check endpoint to verify that the API and model are loaded.

## API Endpoints

### GET `/health`

Checks whether the API is running and the model is loaded.

Example response:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### POST `/predict`

Accepts a question and five answer options.

Request body:

```json
{
  "question": "What is the capital of France?",
  "option_a": "Berlin",
  "option_b": "Madrid",
  "option_c": "Paris",
  "option_d": "Rome",
  "option_e": "London"
}
```

Example response:

```json
{
  "predicted_class": "C",
  "top_3_predictions": [
    {
      "class": "C",
      "probability": 0.95
    },
    {
      "class": "A",
      "probability": 0.03
    },
    {
      "class": "B",
      "probability": 0.02
    }
  ]
}
```

## Run Locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
python -m uvicorn main:app
```

The API will be available at:

`http://127.0.0.1:8000`

Interactive API documentation:

`http://127.0.0.1:8000/docs`
