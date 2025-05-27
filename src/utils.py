import os
import matplotlib.pyplot as plt

def save_plot(fig, name, folder="outputs/figures"):
    """Sauvegarde un graphique matplotlib avec un nom donné dans le dossier spécifié."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}.png")
    fig.savefig(path)
    plt.close(fig)  # pour libérer la mémoire
