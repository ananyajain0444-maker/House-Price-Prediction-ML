import pandas as pd
import numpy as np

def load_data():
    np.random.seed(42)

    data = pd.DataFrame({
        "area": np.random.randint(500, 3000, 200),
        "bedrooms": np.random.randint(1, 5, 200),
        "bathrooms": np.random.randint(1, 4, 200),
        "age": np.random.randint(0, 30, 200),
        "price": np.random.randint(2000000, 15000000, 200)
    })

    return data

def clean_data(df):
    df = df.drop_duplicates()
    df = df.dropna()
    return df