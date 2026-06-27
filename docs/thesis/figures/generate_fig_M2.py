"""
Genera la Figura M2: Muestras representativas del dataset Brain Tumor MRI.
Cuadrícula 2 filas × 4 columnas (una columna por clase).
Fila 1: muestra A de cada clase.
Fila 2: muestra B de cada clase.
Solo imágenes originales (redimensionadas a 224×224 para visualización).

Uso:
    python docs/thesis/figures/generate_fig_M2.py
Salida:
    docs/thesis/figures/fig_M2_brain_tumor_samples.pdf
    docs/thesis/figures/fig_M2_brain_tumor_samples.png
"""

import random
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ── Rutas ────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR   = REPO_ROOT / "data" / "brain_tumor" / "raw" / "Training"
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Clases en el orden de metadata.json ──────────────────────────────────────
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
LABELS  = {
    "glioma":     "Glioma",
    "meningioma": "Meningioma",
    "notumor":    "No Tumor",
    "pituitary":  "Pituitario",
}

# ── Selección reproducible de muestras ───────────────────────────────────────
SEEDS = [7, 42]   # dos muestras visualmente distintas por clase

def pick_sample(class_name: str, seed: int) -> Path:
    images = sorted((DATA_DIR / class_name).glob("*.jpg"))
    random.seed(seed)
    return random.choice(images)

# samples[cls] = [path_muestra_A, path_muestra_B]
samples = {cls: [pick_sample(cls, s) for s in SEEDS] for cls in CLASSES}

# ── Configuración de estilo ───────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":    "serif",
    "font.size":      9,
    "axes.titlesize": 9,
    "axes.titlepad":  5,
})

N_ROWS = 2          # muestra A y muestra B
N_COLS = len(CLASSES)   # una columna por clase

fig_width  = 10.0
fig_height = fig_width * (N_ROWS / N_COLS) * 1.15

fig, axes = plt.subplots(
    N_ROWS, N_COLS,
    figsize=(fig_width, fig_height),
    gridspec_kw={"wspace": 0.06, "hspace": 0.12},
)

# ── Encabezados de columna (nombre de clase) ──────────────────────────────────
for col_idx, cls in enumerate(CLASSES):
    axes[0, col_idx].set_title(
        LABELS[cls],
        fontsize=10,
        fontweight="bold",
        pad=5,
    )

# ── Etiquetas de fila ─────────────────────────────────────────────────────────
row_labels = ["Muestra A", "Muestra B"]

# ── Rellenar celdas ───────────────────────────────────────────────────────────
for row_idx in range(N_ROWS):
    for col_idx, cls in enumerate(CLASSES):
        path = samples[cls][row_idx]
        pil_img = Image.open(path).convert("RGB").resize((224, 224), Image.LANCZOS)
        ax = axes[row_idx, col_idx]
        ax.imshow(np.array(pil_img))
        ax.axis("off")
        # borde sutil
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("#aaaaaa")
            spine.set_linewidth(0.8)

    # Etiqueta a la izquierda de la fila
    axes[row_idx, 0].annotate(
        row_labels[row_idx],
        xy=(0, 0.5),
        xycoords="axes fraction",
        xytext=(-8, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=8,
        color="#444444",
        rotation=90,
    )

# ── Título y pie de figura ────────────────────────────────────────────────────
fig.suptitle(
    "Figura M2: Muestras Representativas del Dataset Brain Tumor MRI",
    fontsize=11,
    fontweight="bold",
    y=1.02,
)

fig.text(
    0.5, -0.02,
    "Fuente: Kaggle Brain Tumor MRI Dataset (Nickparvar, 2021).",
    ha="center",
    fontsize=7,
    style="italic",
    color="gray",
)

# ── Guardar ───────────────────────────────────────────────────────────────────
out_base = OUTPUT_DIR / "fig_M2_brain_tumor_samples"
fig.savefig(str(out_base) + ".pdf", dpi=300, bbox_inches="tight")
fig.savefig(str(out_base) + ".png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("✅ Figura guardada en:")
print(f"   {out_base}.pdf")
print(f"   {out_base}.png")
