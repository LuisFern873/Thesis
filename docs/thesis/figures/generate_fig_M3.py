"""
Genera la Figura M3: Distribución de clases en el conjunto de entrenamiento
(Brain Tumor MRI).

El dataset presenta un desbalance leve en meningioma: la clase cuenta con
1,300 imágenes originales frente a 1,400 en las demás clases.  El proveedor
compensó ese déficit con 100 imágenes aumentadas (prefijo 'Tr-aug-me_'),
lo que da un total nominal de 1,400 por clase. La figura muestra ambas capas
para evidenciar el desbalance original y contextualizar el uso de F1-macro.

Uso:
    python docs/thesis/figures/generate_fig_M3.py
Salida:
    docs/thesis/figures/fig_M3_class_distribution.pdf
    docs/thesis/figures/fig_M3_class_distribution.png
"""

from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Rutas ─────────────────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parent.parent.parent.parent
TRAIN_DIR  = REPO_ROOT / "data" / "brain_tumor" / "raw" / "Training"
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Conteo por clase ───────────────────────────────────────────────────────────
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
LABELS  = ["Glioma", "Meningioma", "No Tumor", "Pituitario"]

counts_orig = []
counts_aug  = []
for cls in CLASSES:
    cls_dir = TRAIN_DIR / cls
    imgs    = list(cls_dir.glob("*.jpg"))
    aug     = sum(1 for f in imgs if "aug" in f.name)
    orig    = len(imgs) - aug
    counts_orig.append(orig)
    counts_aug.append(aug)

counts_orig = np.array(counts_orig)
counts_aug  = np.array(counts_aug)
counts_total = counts_orig + counts_aug

# ── Estilo ─────────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":    "serif",
    "font.size":      11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

COLOR_ORIG = "#4C72B0"   # azul oscuro — imágenes originales
COLOR_AUG  = "#DD8452"   # naranja    — imágenes aumentadas

fig, ax = plt.subplots(figsize=(7, 4.5))

x = np.arange(len(CLASSES))
bar_width = 0.52

# Barras apiladas: originales (base) + aumentadas (encima)
bars_orig = ax.bar(x, counts_orig, bar_width,
                   color=COLOR_ORIG, label="Imágenes originales", zorder=3)
bars_aug  = ax.bar(x, counts_aug,  bar_width,
                   bottom=counts_orig,
                   color=COLOR_AUG,  label="Imágenes aumentadas (data aug.)",
                   zorder=3)

# ── Línea de referencia (máximo por clase) ─────────────────────────────────────
ax.axhline(counts_total.max(), color="gray", linewidth=0.9,
           linestyle="--", zorder=2, label=f"Máximo ({counts_total.max():,})")

# ── Etiquetas de valor encima de cada barra ────────────────────────────────────
for i, (orig, aug, total) in enumerate(zip(counts_orig, counts_aug, counts_total)):
    # total encima
    ax.text(x[i], total + 12, f"{total:,}",
            ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#222222")
    # desglose (orig + aug) si hay aumentadas
    if aug > 0:
        ax.text(x[i], orig / 2, f"{orig:,}",
                ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        ax.text(x[i], orig + aug / 2, f"+{aug}",
                ha="center", va="center", fontsize=7.5, color="white")

# ── Ejes y etiquetas ───────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(LABELS, fontsize=11)
ax.set_ylabel("Número de imágenes", fontsize=11)
ax.set_ylim(0, counts_total.max() * 1.14)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda val, _: f"{int(val):,}"))
ax.tick_params(axis="y", labelsize=9.5)
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#cccccc", zorder=0)

# ── Leyenda ────────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(facecolor=COLOR_ORIG, label="Imágenes originales"),
    mpatches.Patch(facecolor=COLOR_AUG,  label="Imágenes aumentadas"),
    plt.Line2D([0], [0], color="gray", linewidth=0.9, linestyle="--",
               label=f"Máximo por clase ({counts_total.max():,})"),
]
ax.legend(handles=legend_handles, fontsize=9, frameon=True,
          loc="upper right", framealpha=0.9)

# ── Anotación explicativa ──────────────────────────────────────────────────────
ax.annotate(
    "Meningioma requirió\naumentación (+100)",
    xy=(1, counts_orig[1] + counts_aug[1]),
    xytext=(1.55, 1250),
    fontsize=8,
    color="#333333",
    arrowprops=dict(arrowstyle="->", color="#555555", lw=0.9),
    ha="left",
)

# ── Título y pie ───────────────────────────────────────────────────────────────
ax.set_title(
    "Figura M3: Distribución de Clases — Conjunto de Entrenamiento\n"
    "Brain Tumor MRI (Training split, $N=5{,}600$)",
    fontsize=11, fontweight="bold", pad=10,
)

fig.text(
    0.5, -0.04,
    "Fuente: elaboración propia a partir del dataset (Nickparvar, 2021).",
    ha="center", fontsize=8, style="italic", color="gray",
)

# ── Guardar ────────────────────────────────────────────────────────────────────
fig.tight_layout()
out_base = OUTPUT_DIR / "fig_M3_class_distribution"
fig.savefig(str(out_base) + ".pdf", dpi=300, bbox_inches="tight")
fig.savefig(str(out_base) + ".png", dpi=200, bbox_inches="tight")
plt.close(fig)

print("✅ Figura M3 guardada en:")
print(f"   {out_base}.pdf")
print(f"   {out_base}.png")
print()
print("Conteos finales:")
for lbl, orig, aug, tot in zip(LABELS, counts_orig, counts_aug, counts_total):
    mark = " ← desbalance original" if aug > 0 else ""
    print(f"  {lbl:<14} orig={orig:>4}  aug={aug:>3}  total={tot:>4}{mark}")
