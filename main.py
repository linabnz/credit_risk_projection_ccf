import os
import pandas as pd
import numpy as np
import joblib
import argparse
from statsmodels.tsa.filters.hp_filter import hpfilter

from src.visualization import visualiser_predictions
from src.stationarity import (
    tester_stationnarite_macro,
    tester_transformations_ipl,
    tester_stationnarite_segments,
    appliquer_hp_filter_segments,
    tester_stationnarite_hp_segments,
)
from src.modeling import entrainer_modeles_par_segment
from src.scenario_projection import predict_all_models_scenarios

# Étape 0 : Lecture des arguments de la ligne de commande
parser = argparse.ArgumentParser()
parser.add_argument("--modele", type=str, default="RF", choices=["RF", "OLS"])
args = parser.parse_args()

if __name__ == "__main__":

    # Étape 1 : Chargement des données brutes
    segment = pd.read_csv("data/brutes/Données_CCF_PAR_SEGMENT.csv", sep=";")
    macro = pd.read_excel("data/brutes/historique_macro_variables_projet_CCF_FowardLooking_IFRS9.xlsx")

    # Étape 2 : Prétraitement des variables macroéconomiques
    macro["date_dernier_mois"] = pd.to_datetime(macro["date_dernier_mois"], format="%Y-%m")
    macro["cod_prd_ref"] = macro["date_dernier_mois"].dt.year.astype(str) + "T" + macro["date_dernier_mois"].dt.quarter.astype(str)
    macro = macro.drop(columns=["date_dernier_mois"])
    macro = macro[macro["cod_prd_ref"] >= '2009T1']

    # Étape 3 : Calcul de IPL_diff1_hp (composante cyclique avec filtre HP)
    macro["IPL_diff1"] = macro["IPL"].diff()
    macro["IPL_diff1_hp"] = np.nan
    cycle_ipl, _ = hpfilter(macro["IPL_diff1"].dropna(), lamb=1600)
    macro.loc[macro["IPL_diff1"].dropna().index, "IPL_diff1_hp"] = cycle_ipl

    # Étape 4 : Tests de stationnarité et transformation des segments
    tester_stationnarite_macro(macro)
    tester_transformations_ipl(macro)
    tester_stationnarite_segments(segment)
    segment = appliquer_hp_filter_segments(segment, [2, 3])
    tester_stationnarite_hp_segments(segment, [2, 3])

    # Étape 5 : Substitution de la série brute par le cycle HP pour les segments non stationnaires
    segment["Indicateur_moyen_Brut"] = segment.apply(
        lambda row: row["cycle_hp"] if row["note_ref"] in [2, 3] else row["Indicateur_moyen_Brut"], axis=1
    )

    # Étape 6 : Fusion des données segment + macro
    segment["cod_prd_ref"] = segment["cod_prd_ref"].astype(str).str.strip()
    macro["cod_prd_ref"] = macro["cod_prd_ref"].astype(str).str.strip()
    df = pd.merge(segment, macro, on="cod_prd_ref", how="left")
    df = df.drop(columns=['cycle_hp', 'PIB_diff1', 'IPL', 'TCH', 'Inflation', 'IPL_diff1'])

    # Étape 7 : Séparation des données par segment
    segments = {i: df[df["note_ref"] == i].copy() for i in range(1, 6)}

    # Étape 8 : Entraînement des modèles (RF et OLS) et export du résumé
    resume = entrainer_modeles_par_segment(segments)
    print(resume)

    # Étape 9 : Chargement du fichier de scénarios macroéconomiques
    df_scenarios = pd.read_excel("data/Scenario_horizon3ans_propre.xlsx")

    # Étape 10 : Chargement des features sélectionnées par segment
    top_features_dict = {}
    for i in range(1, 6):
        path = f"models/features/selected_features_segment_{i}.pkl"
        try:
            top_features_dict[i] = joblib.load(path)
        except FileNotFoundError:
            print(f"Fichier non trouvé : {path}")

    # Étape 11 : Prédiction des scénarios CENT / PESS / OPT
    results = predict_all_models_scenarios(
        df_raw=df_scenarios,
        top_features_dict=top_features_dict,
        scenarios=["CENT", "PESS", "OPT"]
    )

    # Étape 12 : Export des prédictions dans un fichier CSV par scénario
    os.makedirs("outputs/predictions", exist_ok=True)
    for scenario_name, segment_preds in results.items():
        dfs = []
        for seg, df_pred in segment_preds.items():
            df_pred = df_pred.copy()
            df_pred["segment"] = seg
            dfs.append(df_pred)
        df_all = pd.concat(dfs)
        df_all.to_csv(f"outputs/predictions/predictions_{scenario_name}.csv", index=False)
        print(f"Fichier exporté : outputs/predictions/predictions_{scenario_name}.csv")

    # Étape 13 : Visualisation des prédictions pour chaque segment et scénario
    visualiser_predictions(results, segments, modele=args.modele)
    print("Visualisation des prédictions terminée. Graphiques sauvegardés dans 'outputs/predictions'.")
