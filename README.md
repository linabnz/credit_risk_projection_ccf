# Recalibrage du CCF Forward Looking - Projet MOSEF 2025

##  Contexte du projet

Ce projet s’inscrit dans le cadre du **Master MOSEF – Risque de Crédit** et vise à **modéliser le CCF (Credit Conversion Factor) de manière Forward Looking**. Il répond à une problématique actuelle en lien avec **l’évolution des normes IFRS9** : avec la disparition de la modélisation bâloise pour certains contrats, l’enjeu est d’**anticiper économiquement le comportement du CCF**, comme cela se fait déjà pour la PD ou la LGD.

###  Objectifs

- Projeter le **CCF à horizon 3 ans**, selon différents **scénarios macroéconomiques** (`CENT`, `PESS`, `OPT`).
- Utiliser des données segmentées par `note_ref` (de 1 à 5).
- Comparer deux types de modèles :
  -  Régression linéaire OLS
  -  Random Forest (non-linéaire, robuste)

---

##  Structure du projet

```
.
├── data/                      
├── models/                   # Modèles enregistrés par segment
│   ├── rf/                   # Random Forests
│   ├── ols/                  # Régressions OLS
│   └── features/             # Variables sélectionnées par segment
├── outputs/                  # Visualisations et prédictions
│   └── predictions/          # Graphiques + fichiers CSV des prédictions
├── src/                      # Code source structuré
│   ├── features.py           # Enrichissement des variables macroéconomiques
│   ├── modeling.py           # Entraînement des modèles par segment
│   ├── scenario_projection.py# Prédictions selon scénarios macro
│   ├── stationarity.py       # Tests de stationnarité (ADF, HP Filter, etc.)
│   ├── utils.py              # Fonctions utilitaires (sauvegarde, etc.)
│   └── visualization.py      # Visualisations des projections
├── main.py                   # Script principal d'exécution
├── requirements.txt          # Liste des dépendances Python
└── README.md                 # Ce fichier
```

---

##  Installation & Lancement

### 1. Cloner le dépôt

```bash
git clone https://github.com/lucasvazelle/CREDIT_RISK_PROJECTION_CCF.git
cd CREDIT_RISK_PROJECTION_CCF
```

### 2. Créer et activer un environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate         # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Lancer le script principal

```bash
# Pour exécuter avec Random Forest
python main.py --modele RF

# Pour exécuter avec la régression OLS
python main.py --modele OLS
```

---

##  Ce que fait le script

L'exécution de `main.py` effectue les étapes suivantes :

1. **Tests de stationnarité** sur les variables macroéconomiques (ADF) et les CCF par segment.
2. **Filtrage Hodrick-Prescott** sur les segments non stationnaires (note_ref = 2 et 3).
3. **Fusion des données segmentées avec les variables macro enrichies**.
4. **Sélection automatique des variables explicatives** via RandomForest.
5. **Entraînement des modèles** (RF + OLS) pour chaque segment.
6. **Sauvegarde des modèles et des features utilisées**.
7. **Chargement des scénarios macro (CENT, PESS, OPT)** depuis un fichier Excel.
8. **Prédiction à horizon 3 ans du CCF** pour chaque segment et chaque scénario.
9. **Export des résultats** en CSV + visualisation en PNG.

---

## Méthodologie

### Enrichissement macroéconomique
- Variables dérivées : `diff`, `lag`, `rolling mean`, `interactions`, `quadratiques`, etc.
- Ajout de contextes économiques : `COVID`, `effets post-COVID`, `HP filter`.

### Stationnarité
- Test ADF sur chaque série.
- Filtrage HP pour les séries non stationnaires.

### Modélisation
- **RandomForestRegressor** avec sélection des variables les plus importantes.
- **OLS** avec vérification des hypothèses classiques : DW, Breusch-Pagan, Shapiro, Jarque-Bera.
- Résumé des performances (`R²`, violations des hypothèses, variables utilisées).

### Projection & Visualisation
- Prédiction sur 3 ans via fichier de scénarios.
- Visualisation automatique des résultats (par segment et modèle).

---

## Fichiers de sortie

| Type de fichier         | Chemin                                        |
|-------------------------|-----------------------------------------------|
| Résumé de modélisation  | `models/resume_modelisation.csv`             |
| Prédictions CSV         | `outputs/predictions/predictions_CENT.csv`   |
|                         | `outputs/predictions/predictions_PESS.csv`   |
|                         | `outputs/predictions/predictions_OPT.csv`    |
| Graphiques par segment  | `outputs/predictions/segment_*.png`          |

---



## Auteurs

Projet réalisé par **Sharon Chemmama**  , **Mariam Tarverdian** ,**Lina Benzemma** , **Lucas Vazelle**
Master MOSEF – Université Paris 1  
Promo 2025  


---

