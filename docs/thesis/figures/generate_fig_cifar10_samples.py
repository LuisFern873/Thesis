"""
Genera la figura de muestras representativas de CIFAR-10.
Cuadrícula 2×5: una imagen por clase, con su etiqueta.
Orden de lectura: fila 1 → airplane, automobile, bird, cat, deer
                  fila 2 → dog, frog, horse, ship, truck

Uso:
    python docs/thesis/figures/generate_fig_cifar10_samples.py
Salida:
    docs/thesis/figures/fig_cifar10_samples.pdf
    docs/thesis/figures/fig_cifar10_samples.png
"""

import pickle
import random
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ── Rutas ─────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent.parent.parent
BATCHES_DIR = REPO_ROOT / "data" / "cifar10" / "cifar-10-batches-py"
OUTPUT_DIR  = Path(__file__).resolve().parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Cargar metadatos y un batch de entrenamiento ──────────────────────────────
with open(BATCHES_DIR / "batches.meta", "rb") as f:
    meta = pickle.load(f, encoding="bytes")
CLASS_NAMES = [name.decode() for name in meta[b"label_names"]]

# Cargar data_batch_1 (suficiente para obtener una muestra de cada clase)
with open(BATCHES_DIR / "data_batch_1", "rb") as f:
    batch = pickle.load(f, encoding="bytes")

images_raw = batch[b"data"]          # (10000, 3072) uint8
labels     = batch[b"labels"]        # list of 10000 ints

# Reconstruir imágenes como (N, H, W, C)
images = images_raw.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # (N,32,32,3)

# ── Seleccionar una muestra por clase (reproducible) ─────────────────────────
SEED = 17
random.seed(SEED)

samples = {}   # class_idx → np.ndarray (32,32,3)
for class_idx in range(len(CLASS_NAMES)):
    candidates = [i for i, lbl in enumerate(labels) if lbl == class_idx]
    chosen = random.choice(candidates)
    samples[class_idx] = images[chosen]

# ── Etiquetas en español ──────────────────────────────────────────────────────
LABELS_ES = {
    "airplane":    "Avión",
    "automobile":  "Automóvil",
    "bird":        "Pájaro",
    "cat":         "Gato",
    "deer":        "Ciervo",
    "dog":         "Perro",
    "frog":        "Rana",
    "horse":       "Caballo",
    "ship":        "Barco",
    "truck":       "Camión",
}

# ── Construcción de la figura (2 filas × 5 columnas) ─────────────────────────
matplotlib.rcParams.update({
    "font.family":    "serif",
    "font.size":      10,
    "axes.titlesize": 9.5,
    "axes.titlepad":  5,
})

N_ROWS, N_COLS = 2, 5
fig_width  = 10.0
fig_height = fig_width * (N_ROWS / N_COLS) * 1.10

fig, axes = plt.subplots(
    N_ROWS, N_COLS,
    figsize=(fig_width, fig_height),
    gridspec_kw={"wspace": 0.08, "hspace": 0.30},
)

for idx, class_idx in enumerate(range(len(CLASS_NAMES))):
    row = idx // N_COLS
    col = idx  % N_COLS
    ax  = axes[row, col]

    name_en = CLASS_NAMES[class_idx]
    name_es = LABELS_ES[name_en]

    # Mostrar imagen (32×32, upscaled por matplotlib)
    ax.imshow(samples[class_idx], interpolation="nearest")
    ax.set_title(f"{name_es}\n({name_en})", fontsize=8.5, fontweight="bold", pad=4)
    ax.axis("off")

    # Borde sutil
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#999999")
        spine.set_linewidth(0.7)

# ── Título y pie ──────────────────────────────────────────────────────────────
fig.suptitle(
    "Muestras Representativas del Dataset CIFAR-10\n"
    "(una imagen por clase, resolución original 32×32 px)",
    fontsize=11,
    fontweight="bold",
    y=1.03,
)

fig.text(
    0.5, -0.03,
    "Fuente: CIFAR-10 dataset (Krizhevsky, 2009). Elaboración propia.",
    ha="center",
    fontsize=8,
    style="italic",
    color="gray",
)

# ── Guardar ───────────────────────────────────────────────────────────────────
fig.tight_layout()
out_base = OUTPUT_DIR / "fig_cifar10_samples"
fig.savefig(str(out_base) + ".pdf", dpi=300, bbox_inches="tight")
fig.savefig(str(out_base) + ".png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("✅ Figura CIFAR-10 guardada en:")
print(f"   {out_base}.pdf")
print(f"   {out_base}.png")
