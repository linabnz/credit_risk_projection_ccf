# src/preprocessing.py
import pandas as pd

def convertir_cod_prd_ref_en_date(df):
    df = df.copy()
    df["period"] = df["cod_prd_ref"].str.replace("T", "Q")
    df["period"] = pd.PeriodIndex(df["period"], freq="Q")
    df["date"] = pd.PeriodIndex(df["period"].astype(str), freq="Q").to_timestamp()
    df["year"] = df["period"].dt.year
    df["quarter"] = df["period"].dt.quarter
    return df