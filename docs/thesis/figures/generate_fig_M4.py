"""
Genera la Figura M4: Particiones Dirichlet para Brain Tumor MRI.
Cinco mapas de calor (10 clientes × 4 clases) para α ∈ {0.03, 0.1, 0.3, 1.0, 1000}.
Escala de color: blanco (proporción nula) → rojo oscuro (proporción dominante).
Los datos se leen de los all_stats.json generados por generate_data.py (seed_42).

Uso:
    python docs/thesis/figures/generate_fig_M4.py
Salida:
    docs/thesis/figures/fig_M4_dirichlet_partitions.pdf
    docs/thesis/figures/fig_M4_dirichlet_partitions.png
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Rutas ─────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent.parent.parent.parent
PART_DIR   = REPO_ROOT / "data" / "brain_tumor" / "partitions"
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuración ─────────────────────────────────────────────────────────────
ALPHAS       = [1000.0, 1.0, 0.3, 0.1, 0.03]   # orden: IID → extremo
ALPHA_LABELS = [r"$\alpha=1000$", r"$\alpha=1.0$", r"$\alpha=0.3$",
                r"$\alpha=0.1$", r"$\alpha=0.03$"]
SEED         = "seed_42"
N_CLIENTS    = 10
CLASS_NAMES  = ["Glioma", "Meningioma", "No Tumor", "Pituitario"]
N_CLASSES    = len(CLASS_NAMES)

# ── Leer distribuciones ───────────────────────────────────────────────────────
def load_distribution(alpha: float) -> np.ndarray:
    """Devuelve matriz (N_CLIENTS, N_CLASSES) con proporciones de etiquetas."""
    alpha_str = str(alpha) if alpha != int(alpha) else f"{int(alpha)}.0"
    path = PART_DIR / f"alpha_{alpha_str}" / SEED / "all_stats.json"
    with open(path, "r") as f:
        stats = json.load(f)
    mat = np.zeros((N_CLIENTS, N_CLASSES), dtype=np.float32)
    for client_idx in range(N_CLIENTS):
        dist = stats[str(client_idx)]["label_distribution"]
        mat[client_idx] = dist
    return mat

distributions = [load_distribution(a) for a in ALPHAS]

# ── Estilo ─────────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":    "serif",
    "font.size":      9,
    "axes.titlesize": 9,
    "axes.titlepad":  6,
})

cmap = plt.cm.Reds   # blanco → rojo oscuro

# ── Figura: 5 filas × 1 columna ───────────────────────────────────────────────
# Cada mapa: eje X = clientes (C0…C9), eje Y = clases (4 clases)
# La matriz a visualizar es la TRANSPUESTA de lo leído: (N_CLASSES, N_CLIENTS)

fig, axes = plt.subplots(
    len(ALPHAS), 1,
    figsize=(9.0, 11.0),
    gridspec_kw={"hspace": 0.55},
)

level_map = {1000.0: "IID", 1.0: "Baja", 0.3: "Moderada",
             0.1: "Alta", 0.03: "Extrema"}

ims = []
for ax_idx, (ax, mat, alpha_lbl, alpha_val) in enumerate(
        zip(axes, distributions, ALPHA_LABELS, ALPHAS)):

    # Transponer: filas = clases, columnas = clientes
    mat_T = mat.T  # (N_CLASSES, N_CLIENTS)

    im = ax.imshow(mat_T, cmap=cmap, vmin=0.0, vmax=1.0,
                   aspect="auto", interpolation="nearest")
    ims.append(im)

    # Etiqueta lateral (nivel + alpha)
    ax.set_title(
        f"{alpha_lbl} — {level_map[alpha_val]}",
        fontsize=9, fontweight="bold", loc="left", pad=4,
    )

    # Eje X: clientes
    ax.set_xticks(range(N_CLIENTS))
    if ax_idx == len(ALPHAS) - 1:          # solo en el panel inferior
        ax.set_xticklabels([f"C{k}" for k in range(N_CLIENTS)], fontsize=8)
        ax.set_xlabel("Cliente", fontsize=9)
    else:
        ax.set_xticklabels([])

    # Eje Y: clases
    ax.set_yticks(range(N_CLASSES))
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)

    # Anotar valores en cada celda
    for r in range(N_CLASSES):
        for c in range(N_CLIENTS):
            val = mat_T[r, c]
            if val < 0.005:          # celda prácticamente vacía: no anotar
                continue
            txt_color = "white" if val > 0.55 else "black"
            ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                    fontsize=6.0, color=txt_color, fontweight="bold")

# ── Colorbar global (anclada al último panel, tamaño proporcional) ────────────
fig.subplots_adjust(right=0.83)
cbar_ax = fig.add_axes([0.86, 0.38, 0.018, 0.24])   # centrada verticalmente,
                                                      # altura ≈ un solo panel
cb = fig.colorbar(ims[-1], cax=cbar_ax)
cb.set_label("Proporción de muestras", fontsize=8.5, labelpad=8)
cb.ax.tick_params(labelsize=8)
cb.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

# ── Título y pie ──────────────────────────────────────────────────────────────
fig.suptitle(
    "Figura M4: Particiones Dirichlet — Brain Tumor MRI\n"
    r"($K=10$ clientes, $\alpha \in \{1000,\,1.0,\,0.3,\,0.1,\,0.03\}$, semilla 42)",
    fontsize=10, fontweight="bold", y=1.01,
)

fig.text(
    0.46, -0.01,
    r"Fuente: elaboración propia mediante \texttt{generate\_data.py}.",
    ha="center", fontsize=8, style="italic", color="gray",
)

# ── Guardar ───────────────────────────────────────────────────────────────────
out_base = OUTPUT_DIR / "fig_M4_dirichlet_partitions"
fig.savefig(str(out_base) + ".pdf", dpi=300, bbox_inches="tight")
fig.savefig(str(out_base) + ".png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("✅ Figura M4 guardada en:")
print(f"   {out_base}.pdf")
print(f"   {out_base}.png")
