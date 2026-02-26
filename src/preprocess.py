import pandas as pd


def load_data(path):
    df = pd.read_csv(path)
    return df


def clean_data(df):
    df.columns = df.columns.str.replace(" ", "_")
    df.columns = df.columns.str.strip()

    df["Date"] = pd.to_datetime(df["Date"])
    df["Day"] = df["Date"].dt.day
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year

    df = df.drop(columns=["Date"])

    df = pd.get_dummies(df, drop_first=True)

    return df