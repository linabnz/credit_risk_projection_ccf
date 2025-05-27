# src/stationarity.py
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.filters.hp_filter import hpfilter
from scipy.stats import boxcox

def tester_stationnarite_macro(df_macro, colonnes=None, verbose=True):
    if colonnes is None:
        colonnes = [col for col in df_macro.columns if col != "cod_prd_ref"]
    resultats = {}
    for col in colonnes:
        serie = df_macro[col].dropna()
        try:
            p_value = adfuller(serie, regression='ct')[1]
            resultats[col] = p_value
            if verbose:
                etat = "✅ Stationnaire" if p_value < 0.05 else "❌ Non stationnaire"
                print(f"{col} : p-value = {p_value:.4f} → {etat}")
        except Exception as e:
            resultats[col] = None
            print(f"{col} : ❌ Erreur ADF → {e}")
    return resultats

def tester_transformations_ipl(df, col="IPL_diff1"):
    serie = df[col].dropna()
    resultats = {}
    resultats["Brute"] = adfuller(serie, regression='ct')[1]
    try:
        resultats["Diff(2)"] = adfuller(serie.diff().dropna(), regression='ct')[1]
    except: resultats["Diff(2)"] = None
    try:
        cycle, _ = hpfilter(serie, lamb=1600)
        resultats["HP Cycle"] = adfuller(cycle.dropna(), regression='ct')[1]
    except: resultats["HP Cycle"] = None
    try:
        if (serie > 0).all():
            log_diff = np.log(serie).diff().dropna()
            resultats["Log-Diff"] = adfuller(log_diff, regression='ct')[1]
        else:
            resultats["Log-Diff"] = None
    except: resultats["Log-Diff"] = None
    try:
        if (serie > 0).all():
            bc_trans, _ = boxcox(serie)
            bc_diff = pd.Series(bc_trans).diff().dropna()
            resultats["BoxCox-Diff"] = adfuller(bc_diff, regression='ct')[1]
        else:
            resultats["BoxCox-Diff"] = None
    except: resultats["BoxCox-Diff"] = None
    print("\nRésultats ADF pour différentes transformations :")
    for k, p in resultats.items():
        if p is None:
            print(f"{k}: ❌ Erreur")
        else:
            etat = "✅ Stationnaire" if p < 0.05 else "❌ Non stationnaire"
            print(f"{k}: p-value = {p:.4f} → {etat}")
    return resultats

def tester_stationnarite_segments(segment_df):
    for i in range(1, 6):
        segment_i = segment_df[segment_df["note_ref"] == i].copy()
        try:
            serie = segment_i["Indicateur_moyen_Brut"].str.replace(",", ".").astype(float).dropna()
        except Exception as e:
            print(f"\n❌ Erreur conversion Segment {i} : {e}")
            continue
        print(f"\n🔎 Test ADF - Segment {i}")
        try:
            p_value = adfuller(serie, regression='ct')[1]
            etat = "✅ Stationnaire" if p_value < 0.05 else "❌ Non stationnaire"
            print(f"p-value = {p_value:.4f} → {etat}")
        except Exception as e:
            print(f"⚠️ Erreur ADF pour le segment {i} : {e}")

def appliquer_hp_filter_segments(segment_df, segments_hp):
    for i in segments_hp:
        segment_i = segment_df[segment_df["note_ref"] == i].copy()
        try:
            serie = segment_i["Indicateur_moyen_Brut"].str.replace(",", ".").astype(float).dropna()
            cycle, _ = hpfilter(serie, lamb=1600)
            segment_df.loc[segment_df["note_ref"] == i, "cycle_hp"] = cycle.values
            print(f"✅ HP filter appliqué au segment {i}")
        except Exception as e:
            print(f"❌ Erreur pour le segment {i} : {e}")
    return segment_df

def tester_stationnarite_hp_segments(segment_df, segments_hp):
    for i in segments_hp:
        serie_hp = segment_df.loc[segment_df["note_ref"] == i, "cycle_hp"].dropna()
        print(f"\n🔎 Test ADF sur cycle HP - Segment {i}")
        if len(serie_hp) < 10:
            print("⚠️ Trop peu de données pour appliquer ADF")
            continue
        try:
            p_value = adfuller(serie_hp, regression="ct")[1]
            etat = "✅ Stationnaire" if p_value < 0.05 else "❌ Non stationnaire"
            print(f"p-value = {p_value:.4f} → {etat}")
        except Exception as e:
            print(f"❌ Erreur ADF pour le segment {i} : {e}")