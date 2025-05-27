import os
import matplotlib.pyplot as plt
import pandas as pd

def visualiser_predictions(results_scenarios, segments, modele="RF"):
    os.makedirs("outputs/predictions", exist_ok=True)
    couleurs = {"CENT": "blue", "PESS": "red", "OPT": "green"}

    if isinstance(segments, list):
        segments_dict = {i + 1: df.copy() for i, df in enumerate(segments)}
    else:
        segments_dict = segments

    for seg in range(1, 6):
        fig, ax = plt.subplots(figsize=(12, 5))
        df_hist = segments_dict[seg].copy()
        df_hist["Indicateur_moyen_Brut"] = (
            df_hist["Indicateur_moyen_Brut"]
            .astype(str).str.replace(",", ".").astype(float)
        )
        df_hist = df_hist.dropna(subset=["Indicateur_moyen_Brut"]).reset_index(drop=True)
        if "date" not in df_hist.columns:
            nb_hist = len(df_hist)
            df_hist["date"] = pd.date_range(end="2023-12-31", periods=nb_hist, freq="QE-DEC")
        df_hist["trimestre"] = pd.to_datetime(df_hist["date"]).dt.to_period("Q").astype(str)
        df_hist = df_hist.sort_values("date")

        ax.plot(df_hist["trimestre"],
                df_hist["Indicateur_moyen_Brut"],
                label="Historique réel", marker="o", linestyle="--", color="black")

        for scenario in ["CENT", "PESS", "OPT"]:
            try:
                df_pred = results_scenarios[scenario][seg].copy()
                df_pred["date"] = pd.to_datetime(df_pred["date"])
                df_pred["trimestre"] = df_pred["date"].dt.to_period("Q").astype(str)
                df_pred = df_pred.sort_values("date")

                y_col = f"CCF_{modele.upper()}"
                if y_col not in df_pred.columns:
                    raise ValueError(f"Colonne {y_col} non trouvée.")

                ax.plot(df_pred["trimestre"], df_pred[y_col],
                        label=f"{scenario} – {modele.upper()}",
                        marker="x", linestyle="-", color=couleurs.get(scenario, None))
            except Exception as e:
                print(f"Segment {seg} – Scénario {scenario} erreur : {e}")

        ax.set_title(f"Segment {seg} – CCF ({modele.upper()}) : Réel + Prédictions")
        ax.set_xlabel("Trimestre")
        ax.set_ylabel("CCF")
        ax.grid(True)
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        path = f"outputs/predictions/segment_{seg}_predictions_{modele.upper()}.png"
        plt.savefig(path)
        plt.close()
        print(f"Figure enregistrée : {path}")
