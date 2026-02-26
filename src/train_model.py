import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocess import load_data, clean_data

DATA_PATH = "data/sales_data.csv"
MODEL_PATH = "models/demand_model.pkl"


def train():
    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Cleaning data...")
    df = clean_data(df)

    # Define features and target
    X = df.drop(columns=["Units_Sold"])
    y = df["Units_Sold"]

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training model...")

    model = RandomForestRegressor(
        n_estimators=20,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2 Score:", r2)

    # ---------------------------
    # Feature Importance Graph
    # ---------------------------
    print("Generating Feature Importance Graph...")

    importances = model.feature_importances_
    feature_names = X.columns

    sorted_features = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    features = [f[0] for f in sorted_features]
    scores = [f[1] for f in sorted_features]

    plt.figure()
    plt.barh(features[::-1], scores[::-1])
    plt.xlabel("Importance Score")
    plt.title("Top 10 Feature Importances")
    plt.tight_layout()
    plt.show()

    # ---------------------------
    # Save Model
    # ---------------------------
    print("Saving model...")

    # Create models folder if it doesn't exist
    os.makedirs("models", exist_ok=True)

    joblib.dump({
        "model": model,
        "features": X.columns.tolist()
    }, MODEL_PATH)

    print("Model saved successfully at models/demand_model.pkl")


if __name__ == "__main__":
    train()