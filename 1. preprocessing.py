import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def clean_data(df):
    return df.dropna().drop_duplicates()

def normalize_data(df):
    scaler = StandardScaler()
    df[['Humidity', 'Temperature']] = scaler.fit_transform(df[['Humidity', 'Temperature']])
    return df, scaler

def encode_labels(labels):
    le = LabelEncoder()
    return le.fit_transform(labels), le
