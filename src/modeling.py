import os
import pandas as pd
import numpy as np
import joblib
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import shapiro, jarque_bera

from src.preprocessing import convertir_cod_prd_ref_en_date
from src.features import enrichir_variables_macro

def entrainer_modeles_par_segment(segments_dict, output_dir="models"):
    os.makedirs(os.path.join(output_dir, "rf"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "ols"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "features"), exist_ok=True)

    resume = []

    for i, df_seg in segments_dict.items():
        try:
            df_seg = convertir_cod_prd_ref_en_date(df_seg)
            df_enrichi = enrichir_variables_macro(df_seg)
            df_enrichi["Indicateur_moyen_Brut"] = (
                df_enrichi["Indicateur_moyen_Brut"]
                .astype(str).str.replace(",", ".").astype(float)
            )

            X = df_enrichi.drop(columns=["date", "Indicateur_moyen_Brut"], errors="ignore")
            y = df_enrichi["Indicateur_moyen_Brut"]
            X = X.select_dtypes(include=[np.number]).fillna(0)

            selector = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=0))
            selector.fit(X, y)
            top_vars = list(X.columns[selector.get_support()])
            X_sel = df_enrichi[top_vars].fillna(0)

            joblib.dump(top_vars, os.path.join(output_dir, "features", f"selected_features_segment_{i}.pkl"))

            model_rf = RandomForestRegressor(n_estimators=100, random_state=0).fit(X_sel, y)
            r2_rf = model_rf.score(X_sel, y)
            joblib.dump((model_rf, top_vars), os.path.join(output_dir, "rf", f"segment_{i}.joblib"))

            X_ols = sm.add_constant(X_sel)
            model_ols = sm.OLS(y, X_ols).fit()
            r2_ols = model_ols.rsquared
            joblib.dump((model_ols, top_vars), os.path.join(output_dir, "ols", f"segment_{i}.joblib"))

            dw = durbin_watson(model_ols.resid)
            bp_p = het_breuschpagan(model_ols.resid, X_ols)[1]
            shap_p = shapiro(model_ols.resid)[1]
            jb_p = jarque_bera(model_ols.resid)[1]

            violations = []
            if not (1.5 <= dw <= 2.5): violations.append("DW")
            if bp_p <= 0.05: violations.append("BP")
            if shap_p <= 0.05: violations.append("Shapiro")
            if jb_p <= 0.05: violations.append("JB")

            resume.append({
                "Segment": i,
                "R2_RF": round(r2_rf, 3),
                "R2_OLS": round(r2_ols, 3),
                "Hypothèses non respectées": ", ".join(violations) if violations else "Aucune",
                "Variables utilisées": ", ".join(top_vars)
            })

        except Exception as e:
            print(f"Erreur segment {i} : {e}")

    df_resume = pd.DataFrame(resume)
    os.makedirs(output_dir, exist_ok=True)
    df_resume.to_csv(os.path.join(output_dir, "resume_modelisation.csv"), index=False)
    return df_resume
