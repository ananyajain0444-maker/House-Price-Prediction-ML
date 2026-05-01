def feature_engineering(df):
    df["price_per_sqft"] = df["price"] / df["area"]
    return df