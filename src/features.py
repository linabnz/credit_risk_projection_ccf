
# src/features.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

def enrichir_variables_macro(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["PIB_lag1"] = df["PIB"].shift(1)
    df["TCH_diff1_lag1"] = df["TCH_diff1"].shift(1)
    df["Inflation_diff1_lag1"] = df["Inflation_diff1"].shift(1)
    df["IPL_diff1_hp_lag1"] = df["IPL_diff1_hp"].shift(1)
    df["PIB_lag2"] = df["PIB"].shift(2)
    df["TCH_diff1_lag2"] = df["TCH_diff1"].shift(2)
    df["Inflation_diff1_lag2"] = df["Inflation_diff1"].shift(2)
    df["IPL_diff1_hp_lag2"] = df["IPL_diff1_hp"].shift(2)
    df["PIB_ma3"] = df["PIB"].rolling(window=3).mean()
    df["TCH_ma3"] = df["TCH_diff1"].rolling(window=3).mean()
    df["Inflation_ma3"] = df["Inflation_diff1"].rolling(window=3).mean()
    df["IPL_ma3"] = df["IPL_diff1_hp"].rolling(window=3).mean()
    df["PIB_ma5"] = df["PIB"].rolling(window=5).mean()
    df["TCH_ma5"] = df["TCH_diff1"].rolling(window=5).mean()
    df["Inflation_ma5"] = df["Inflation_diff1"].rolling(window=5).mean()
    df["IPL_ma5"] = df["IPL_diff1_hp"].rolling(window=5).mean()
    df["PIB_x_TCH"] = df["PIB"] * df["TCH_diff1"]
    df["PIB_x_Inflation"] = df["PIB"] * df["Inflation_diff1"]
    df["TCH_x_IPL"] = df["TCH_diff1"] * df["IPL_diff1_hp"]
    df["Inflation_x_IPL"] = df["Inflation_diff1"] * df["IPL_diff1_hp"]
    df["PIB_x_TCH_ma3"] = df["PIB"] * df["TCH_diff1"].rolling(3).mean()
    df["PIB_squared"] = df["PIB"] ** 2
    df["TCH_diff1_squared"] = df["TCH_diff1"] ** 2
    df["Inflation_diff1_squared"] = df["Inflation_diff1"] ** 2
    df["IPL_diff1_hp_squared"] = df["IPL_diff1_hp"] ** 2
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter
    df["is_covid"] = ((df["date"] >= "2020-03-01") & (df["date"] <= "2021-06-30")).astype(int)
    df["post_covid"] = (df["date"] >= "2021-07-01").astype(int)
    df["PIB_pct_change"] = df["PIB"].pct_change()
    df["TCH_diff1_abs"] = df["TCH_diff1"].abs()
    df["PIB_x_TCH_squared"] = df["PIB"] * (df["TCH_diff1"] ** 2)


    df = df.dropna().reset_index(drop=True)
    print(f"🧪 Données enrichies : {df.shape[1]} variables disponibles.")
    return df

def select_features_via_random_forest(df, target_col="Indicateur_moyen_Brut", n_estimators=100):
    X = df.drop(columns=["date", target_col])
    y = df[target_col].astype(str).str.replace(",", ".").astype(float)
    X = X.select_dtypes(include=[np.number]).fillna(0)
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=0)
    selector = SelectFromModel(model).fit(X, y)
    selected_vars = list(X.columns[selector.get_support()])
    print(f"🎯 Variables sélectionnées ({len(selected_vars)}):", selected_vars)
    return selected_vars
