Here’s a README.md you can use for your project:

# XGBoost Stock Price Prediction API

This project is a machine learning-powered REST API built with Flask that predicts stock prices using the **XGBoost Regressor**. It accepts historical stock data in CSV or JSON format, trains a model on `Adj Close` prices, and supports **batch predictions**.

---

## Features

- **Machine Learning Model**: Uses `XGBoostRegressor` with a preprocessing pipeline including `StandardScaler`.
- **Batch Prediction**: Accepts multiple rows of stock data in JSON or CSV and returns predictions for each.
- **REST API**: Exposes endpoints for uploading data, training the model, and making predictions.

---

## API Endpoints

### `POST /upload`

Upload a CSV file containing stock data.

**Request**: `multipart/form-data`  
**Form Field**: `file` = your `.csv` file  
**Columns Required**:

Date,Open,High,Low,Close,Adj Close,Volume

---

### `POST /train`

Train the model using the uploaded dataset. It uses `Adj Close` as the target label.

**Response**:
```json
{
  "message": "Model trained and saved",
  "test_metrics": {
    "MAE": 1.23,
    "R2": 0.94
  }
}


---

POST /predict

Make batch predictions using the trained model.

Request (JSON):

[
  {
    "Date": "2014-09-29",
    "Open": 100.58,
    "High": 100.69,
    "Low": 98.04,
    "Close": 99.62,
    "Adj Close": 93.51,
    "Volume": 142718700
  },
  ...
]

Response:

{
  "prediction": [93.92, 94.50, 91.83, ...]
}


---

How to Run Locally

1. Install requirements:



pip install flask pandas scikit-learn xgboost joblib

2. Start the server:



python server.py


---

Files

model.py: Contains model logic, training, and batch prediction support.

server.py: Flask API server.

uploaded_data.csv: Sample dataset for training.

xgb_model.pkl: Saved XGBoost model.

synthetic_stock_data.csv: Optional example dataset.



---

Notes

Model automatically extracts date features: Year, Month, Day, DayOfWeek.

File must include "Adj Close" column for training.

You can retrain anytime by re-uploading new data and calling /train.



---

Author

GitHub: cecepmakaveli


