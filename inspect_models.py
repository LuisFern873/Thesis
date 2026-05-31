"""
inspect_models.py
=================
Genera un informe detallado de la estructura de cada arquitectura registrada
en MODELS (src/utils/models.py).

Para cada modelo muestra:
  - Nombre y tipo de cada capa (en orden de ejecución)
  - Parámetros relevantes de la capa (canales, kernel, grupos, etc.)
  - Clasificación según classify_layer() de drift_metrics.py
  - Resumen agrupado por clasificación (norm / feature / head / other)
  - Conteo de parámetros totales y por grupo

Uso
---
    # Informe de todos los modelos (dataset por defecto: cifar10)
    python inspect_models.py

    # Informe de modelos específicos
    python inspect_models.py --models efficient0 efficient0_gn vit_tiny

    # Cambiar dataset de referencia
    python inspect_models.py --dataset brain_tumor

    # Guardar salida en fichero
    python inspect_models.py --output reports/model_report.txt

    # Exportar a JSON
    python inspect_models.py --json reports/model_report.json

    # Omitir modelos que fallen al importar (ej. vim_tiny sin mamba_ssm)
    python inspect_models.py --skip-errors
"""

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root without installing the package
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.models import MODELS
from src.utils.drift_metrics import classify_layer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Default dataset used to instantiate models (must exist in NUM_CLASSES /
# INPUT_CHANNELS).  cifar10 is safe for all architectures.
DEFAULT_DATASET = "cifar10"

# Rich-style separators
SEP_THICK = "=" * 90
SEP_THIN  = "-" * 90


def _layer_params(module: nn.Module) -> dict[str, Any]:
    """Extract the most informative constructor-level attributes of a layer."""
    info: dict[str, Any] = {}
    cls = type(module)

    # --- Convolutions ---
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        info["in_channels"]  = module.in_channels
        info["out_channels"] = module.out_channels
        info["kernel_size"]  = module.kernel_size
        info["stride"]       = module.stride
        info["padding"]      = module.padding
        info["groups"]       = module.groups
        info["bias"]         = module.bias is not None

    # --- Linear ---
    elif isinstance(module, nn.Linear):
        info["in_features"]  = module.in_features
        info["out_features"] = module.out_features
        info["bias"]         = module.bias is not None

    # --- Normalization ---
    elif isinstance(module, nn.BatchNorm2d):
        info["num_features"] = module.num_features
        info["eps"]          = module.eps
        info["momentum"]     = module.momentum
        info["affine"]       = module.affine
        info["track_running_stats"] = module.track_running_stats

    elif isinstance(module, nn.BatchNorm1d):
        info["num_features"] = module.num_features
        info["affine"]       = module.affine

    elif isinstance(module, nn.GroupNorm):
        info["num_groups"]   = module.num_groups
        info["num_channels"] = module.num_channels
        info["eps"]          = module.eps
        info["affine"]       = module.affine

    elif isinstance(module, nn.LayerNorm):
        info["normalized_shape"] = list(module.normalized_shape)
        info["eps"]              = module.eps
        info["elementwise_affine"] = module.elementwise_affine

    elif isinstance(module, nn.InstanceNorm2d):
        info["num_features"] = module.num_features
        info["affine"]       = module.affine

    # --- Pooling ---
    elif isinstance(module, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)):
        info["kernel_size"] = module.kernel_size
        info["stride"]      = module.stride
        info["padding"]     = module.padding

    elif isinstance(module, (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)):
        info["kernel_size"] = module.kernel_size
        info["stride"]      = module.stride

    elif isinstance(module, nn.AdaptiveAvgPool2d):
        info["output_size"] = module.output_size

    # --- Dropout ---
    elif isinstance(module, (nn.Dropout, nn.Dropout2d)):
        info["p"] = module.p

    # --- Embedding ---
    elif isinstance(module, nn.Embedding):
        info["num_embeddings"] = module.num_embeddings
        info["embedding_dim"]  = module.embedding_dim

    # --- MultiheadAttention ---
    elif isinstance(module, nn.MultiheadAttention):
        info["embed_dim"]   = module.embed_dim
        info["num_heads"]   = module.num_heads
        info["dropout"]     = module.dropout
        info["batch_first"] = module.batch_first

    return info


def _count_params(module: nn.Module) -> int:
    """Total number of parameters (trainable + frozen) in a module."""
    return sum(p.numel() for p in module.parameters())


def _count_trainable(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _fmt_params(n: int) -> str:
    """Human-readable parameter count: 1_234_567 → '1.23 M'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f} M"
    if n >= 1_000:
        return f"{n / 1_000:.1f} K"
    return str(n)


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyse_model(model_key: str, model: nn.Module) -> dict:
    """
    Walk every named module of *model* and collect per-layer metadata.

    Returns a dict with:
        model_key, class_name, total_params, trainable_params,
        layers: list of layer-dicts,
        summary: {group: {count, params}} per classification group
    """
    layers = []

    for name, module in model.named_modules():
        # Skip the root module and pure containers (Sequential, ModuleList…)
        # that have no own parameters beyond their children.
        if name == "":
            continue
        children = list(module.children())
        is_leaf = len(children) == 0

        # For non-leaf containers we still record them but mark as container.
        classification = classify_layer(name, module)
        own_params = sum(
            p.numel() for p in module.parameters(recurse=False)
        )
        total_params_in_subtree = _count_params(module)

        layer_info = {
            "name":           name,
            "type":           type(module).__name__,
            "is_leaf":        is_leaf,
            "classification": classification,
            "own_params":     own_params,
            "subtree_params": total_params_in_subtree,
            "layer_params":   _layer_params(module),
        }
        layers.append(layer_info)

    # --- Summary by classification group (leaf layers only) ---
    groups = ["norm", "feature", "head", "other"]
    summary: dict[str, dict] = {g: {"count": 0, "params": 0} for g in groups}
    for lyr in layers:
        if not lyr["is_leaf"]:
            continue
        g = lyr["classification"]
        if g not in summary:
            summary[g] = {"count": 0, "params": 0}
        summary[g]["count"]  += 1
        summary[g]["params"] += lyr["own_params"]

    total_p     = _count_params(model)
    trainable_p = _count_trainable(model)

    return {
        "model_key":        model_key,
        "class_name":       type(model).__name__,
        "total_params":     total_p,
        "trainable_params": trainable_p,
        "layers":           layers,
        "summary":          summary,
    }


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------

_GROUP_COLORS = {
    "norm":    "🟡",
    "feature": "🔵",
    "head":    "🔴",
    "other":   "⚪",
}

_COL_W = {
    "name":  42,
    "type":  26,
    "cls":    8,
    "params": 10,
    "detail": 0,   # fills remaining width
}


def _render_layer_params(lp: dict) -> str:
    if not lp:
        return ""
    parts = []
    for k, v in lp.items():
        parts.append(f"{k}={v}")
    return "  [" + ", ".join(parts) + "]"


def render_text_report(analysis: dict, show_containers: bool = False) -> str:
    lines = []
    key   = analysis["model_key"]
    cls   = analysis["class_name"]
    tp    = analysis["total_params"]
    trp   = analysis["trainable_params"]

    lines.append(SEP_THICK)
    lines.append(f"  MODEL: {key}  ({cls})")
    lines.append(f"  Total params: {_fmt_params(tp)}  |  Trainable: {_fmt_params(trp)}")
    lines.append(SEP_THICK)

    # --- Column header ---
    header = (
        f"  {'Layer name':<{_COL_W['name']}} "
        f"{'Type':<{_COL_W['type']}} "
        f"{'Group':<{_COL_W['cls']}} "
        f"{'Params':>{_COL_W['params']}}  "
        f"Details"
    )
    lines.append(header)
    lines.append(SEP_THIN)

    for lyr in analysis["layers"]:
        if not show_containers and not lyr["is_leaf"]:
            continue

        icon  = _GROUP_COLORS.get(lyr["classification"], "⚪")
        name  = lyr["name"]
        # Truncate very long names
        if len(name) > _COL_W["name"] - 1:
            name = "…" + name[-((_COL_W["name"] - 2)):]

        params_str = _fmt_params(lyr["own_params"]) if lyr["own_params"] > 0 else "-"
        detail_str = _render_layer_params(lyr["layer_params"])

        line = (
            f"  {name:<{_COL_W['name']}} "
            f"{lyr['type']:<{_COL_W['type']}} "
            f"{icon}{lyr['classification']:<{_COL_W['cls'] - 1}} "
            f"{params_str:>{_COL_W['params']}}  "
            f"{detail_str}"
        )
        lines.append(line)

    # --- Group summary ---
    lines.append(SEP_THIN)
    lines.append("  GROUP SUMMARY (leaf layers only)")
    lines.append(SEP_THIN)
    lines.append(
        f"  {'Group':<12} {'Layers':>8} {'Params':>12} {'% of total':>12}"
    )
    lines.append("  " + "-" * 48)
    for g, data in analysis["summary"].items():
        pct = (data["params"] / tp * 100) if tp > 0 else 0.0
        icon = _GROUP_COLORS.get(g, "⚪")
        lines.append(
            f"  {icon}{g:<11} {data['count']:>8} "
            f"{_fmt_params(data['params']):>12} "
            f"{pct:>11.1f}%"
        )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def build_json_report(analyses: list[dict]) -> list[dict]:
    """Serialise analyses to a JSON-safe structure."""
    out = []
    for a in analyses:
        entry = {
            "model_key":        a["model_key"],
            "class_name":       a["class_name"],
            "total_params":     a["total_params"],
            "trainable_params": a["trainable_params"],
            "summary": {
                g: {
                    "count":  v["count"],
                    "params": v["params"],
                    "pct_of_total": round(
                        v["params"] / a["total_params"] * 100, 2
                    ) if a["total_params"] > 0 else 0.0,
                }
                for g, v in a["summary"].items()
            },
            "layers": [
                {
                    "name":           lyr["name"],
                    "type":           lyr["type"],
                    "is_leaf":        lyr["is_leaf"],
                    "classification": lyr["classification"],
                    "own_params":     lyr["own_params"],
                    "layer_params":   lyr["layer_params"],
                }
                for lyr in a["layers"]
            ],
        }
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Model instantiation
# ---------------------------------------------------------------------------

def _try_instantiate(key: str, dataset: str, skip_errors: bool) -> nn.Module | None:
    """Attempt to build a model; return None on failure if skip_errors=True."""
    factory = MODELS[key]
    try:
        model = factory(dataset=dataset, pretrained=False)
        model.eval()
        return model
    except Exception as exc:
        msg = f"[SKIP] {key}: {type(exc).__name__}: {exc}"
        if skip_errors:
            print(msg, file=sys.stderr)
            return None
        raise RuntimeError(msg) from exc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inspect all FL-bench model architectures and classify their layers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(__doc__),
    )
    p.add_argument(
        "--models", nargs="*", metavar="KEY",
        help="Subset of model keys to inspect (default: all in MODELS).",
    )
    p.add_argument(
        "--dataset", default=DEFAULT_DATASET,
        help=f"Dataset name used to instantiate models (default: {DEFAULT_DATASET}).",
    )
    p.add_argument(
        "--output", metavar="FILE",
        help="Write text report to FILE instead of stdout.",
    )
    p.add_argument(
        "--json", metavar="FILE",
        help="Also export a JSON report to FILE.",
    )
    p.add_argument(
        "--show-containers", action="store_true",
        help="Include non-leaf container modules (Sequential, etc.) in the layer table.",
    )
    p.add_argument(
        "--skip-errors", action="store_true",
        help="Skip models that fail to instantiate (e.g. vim_tiny without mamba_ssm).",
    )
    p.add_argument(
        "--list", action="store_true",
        help="Just print all registered model keys and exit.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.list:
        print("Registered models in MODELS:")
        for k in sorted(MODELS):
            print(f"  {k}")
        return

    keys_to_run = args.models if args.models else list(MODELS.keys())

    # Validate requested keys
    unknown = [k for k in keys_to_run if k not in MODELS]
    if unknown:
        print(f"ERROR: Unknown model key(s): {unknown}", file=sys.stderr)
        print(f"Available: {sorted(MODELS.keys())}", file=sys.stderr)
        sys.exit(1)

    text_lines: list[str] = []
    analyses:   list[dict] = []

    header = (
        f"\n{'#' * 90}\n"
        f"  FL-bench Model Architecture Inspector\n"
        f"  Dataset: {args.dataset}  |  Models: {len(keys_to_run)}\n"
        f"{'#' * 90}\n"
    )
    text_lines.append(header)
    print(header, end="")

    for key in keys_to_run:
        print(f"  Analysing {key} ...", end=" ", flush=True)
        model = _try_instantiate(key, args.dataset, args.skip_errors)
        if model is None:
            print("SKIPPED")
            continue

        analysis = analyse_model(key, model)
        analyses.append(analysis)

        report = render_text_report(analysis, show_containers=args.show_containers)
        text_lines.append(report)
        print("OK")

    # --- Cross-model comparison table ---
    if len(analyses) > 1:
        comp_lines = [
            "\n" + SEP_THICK,
            "  CROSS-MODEL COMPARISON",
            SEP_THICK,
            f"  {'Model':<20} {'Total params':>14} "
            f"{'norm %':>8} {'feature %':>10} {'head %':>8} {'other %':>8}",
            "  " + "-" * 72,
        ]
        for a in analyses:
            tp = a["total_params"]
            def pct(g):
                return (
                    a["summary"].get(g, {}).get("params", 0) / tp * 100
                    if tp > 0 else 0.0
                )
            comp_lines.append(
                f"  {a['model_key']:<20} {_fmt_params(tp):>14} "
                f"{pct('norm'):>7.1f}% "
                f"{pct('feature'):>9.1f}% "
                f"{pct('head'):>7.1f}% "
                f"{pct('other'):>7.1f}%"
            )
        comp_lines.append("")
        comp_text = "\n".join(comp_lines)
        text_lines.append(comp_text)
        print(comp_text)

    full_text = "\n".join(text_lines)

    # --- Write text output ---
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(full_text, encoding="utf-8")
        print(f"\nText report saved → {out_path}")
    else:
        # Print individual model reports (comparison already printed above)
        for a in analyses:
            print(render_text_report(a, show_containers=args.show_containers))

    # --- Write JSON output ---
    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_data = build_json_report(analyses)
        json_path.write_text(
            json.dumps(json_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"JSON report saved → {json_path}")


if __name__ == "__main__":
    main()
