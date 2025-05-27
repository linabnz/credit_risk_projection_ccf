# src/scenario_projection.py
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.filters.hp_filter import hpfilter
from src.utils import save_plot
from src.features import enrichir_variables_macro



def prepare_scenario(df_raw, prefix):
    df = df_raw[["date", f"PIB_{prefix}", f"IPL_{prefix}", f"TCH_{prefix}", f"Inflation_{prefix}"]].copy()
    df.columns = ["date", "PIB", "IPL", "TCH", "Inflation"]
    df["PIB_diff1"] = df["PIB"].diff()
    df["TCH_diff1"] = df["TCH"].diff()
    df["Inflation_diff1"] = df["Inflation"].diff()
    df["IPL_diff1"] = df["IPL"].diff()
    df["IPL_diff1_hp"] = np.nan
    cycle_ipl, _ = hpfilter(df["IPL_diff1"].dropna(), lamb=1600)
    df.loc[df["IPL_diff1"].dropna().index, "IPL_diff1_hp"] = cycle_ipl
    return df.dropna().reset_index(drop=True)


def enrichir_macro_scenario(df):
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
    return df.dropna().reset_index(drop=True)


def predict_all_models_scenarios(*, df_raw, top_features_dict, scenarios=["CENT", "PESS", "OPT"], model_dir="models"):
    results = {}
    for scenario in scenarios:
        print(f"\n🔮 Scénario : {scenario}")
        df_prepared = prepare_scenario(df_raw, scenario)
        df_enriched = enrichir_variables_macro(df_prepared)

        scenario_results = {}
        for seg in range(1, 6):
            try:
                model_rf, features_rf = joblib.load(os.path.join(model_dir, "rf", f"segment_{seg}.joblib"))
                model_ols, features_ols = joblib.load(os.path.join(model_dir, "ols", f"segment_{seg}.joblib"))

                # 🔁 Chargement automatique des features sélectionnées
                features_selected = joblib.load(os.path.join(model_dir, "features", f"selected_features_segment_{seg}.pkl"))

                # Filtrage des features pour RF et OLS
                X_rf = df_enriched[features_rf].astype(float).dropna()
                X_ols = df_enriched[features_ols].astype(float).dropna()
                X_ols_const = sm.add_constant(X_ols)

                # Prédictions
                y_pred_rf = model_rf.predict(X_rf)
                y_pred_ols = model_ols.predict(X_ols_const)

                idx_common = X_rf.index.intersection(X_ols.index)
                df_result = df_enriched.loc[idx_common].copy()
                df_result["CCF_RF"] = y_pred_rf[:len(idx_common)]
                df_result["CCF_OLS"] = y_pred_ols[:len(idx_common)]
                scenario_results[seg] = df_result

                print(f"✅ Segment {seg} – {len(idx_common)} prédictions")

                # === 🔽 PLOT ET SAUVEGARDE 🔽 ===
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(df_result["date"], df_result["CCF_RF"], label="RF", linestyle="-", marker="x")
                ax.plot(df_result["date"], df_result["CCF_OLS"], label="OLS", linestyle="--", marker="o")
                ax.set_title(f"{scenario} – Segment {seg} – CCF projeté (RF vs OLS)")
                ax.set_xlabel("Date")
                ax.set_ylabel("CCF prédite")
                ax.grid(True)
                ax.legend()
                plt.tight_layout()
                save_plot(fig, name=f"{scenario}_Segment_{seg}_predictions")

            except Exception as e:
                print(f"❌ Segment {seg} – erreur : {e}")
        results[scenario] = scenario_results
    return results