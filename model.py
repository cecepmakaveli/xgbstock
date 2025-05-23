# model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

class LinearModel:
    def __init__(self):
        self.model = None
        self.model_path = "xgb_model.pkl"

    def load_data(self, file_path="uploaded_data.csv"):
        df = pd.read_csv(file_path, parse_dates=["Date"])

        # Feature engineering from Date
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Day"] = df["Date"].dt.day
        df["DayOfWeek"] = df["Date"].dt.dayofweek

        # Drop unnecessary columns
        df = df.drop(columns=["Date", "Close"])  # Use "Adj Close" as y

        X = df.drop("Adj Close", axis=1)
        y = df["Adj Close"]
        return X, y

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1))
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        self.model = pipeline
        joblib.dump(self.model, self.model_path)

        return {
            "MAE": mean_absolute_error(y_test, y_pred),
            "R2": r2_score(y_test, y_pred)
        }

    def predict(self, input_df):
        input_df = input_df.copy()
        input_df["Date"] = pd.to_datetime(input_df["Date"])
        input_df["Year"] = input_df["Date"].dt.year
        input_df["Month"] = input_df["Date"].dt.month
        input_df["Day"] = input_df["Date"].dt.day
        input_df["DayOfWeek"] = input_df["Date"].dt.dayofweek
        input_df = input_df.drop(columns=["Date", "Close"])  # Drop Close to match training

        if self.model is None:
            self.model = joblib.load(self.model_path)

        prediction = self.model.predict(input_df)
        return prediction.tolist()
