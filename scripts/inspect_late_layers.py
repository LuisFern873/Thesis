"""
inspect_late_layers.py
======================
Muestra qué capas de un modelo serán clasificadas como EARLY o LATE
por FedLateGram, usando el mismo criterio de _detect_late_layers()
del servidor (fallback: últimos `late_fraction` de sub-módulos de base
+ classifier).

Uso:
    python scripts/inspect_late_layers.py --model vim_tiny --late_fraction 0.4
    python scripts/inspect_late_layers.py --model vit_tiny --late_fraction 0.4
    python scripts/inspect_late_layers.py --model vig_tiny --late_fraction 0.4
    python scripts/inspect_late_layers.py --model res9     --late_fraction 0.4
    python scripts/inspect_late_layers.py --all            --late_fraction 0.4

Flags:
    --model          Nombre del modelo en el registro MODELS (requerido, o usar --all)
    --all            Analizar vim_tiny, vig_tiny, vit_tiny y res9 de una vez
    --dataset        Dataset para determinar num_classes (default: cifar10)
    --late_fraction  Fracción de sub-módulos de base tratados como LATE (default: 0.4)
    --pretrained     Usar pesos preentrenados (default: False)
    --show_params    Muestra también los parámetros individuales de cada módulo
"""

import argparse
import sys
import os

# ── Root en sys.path ──────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn as nn

# ── Intenta importar MODELS; si falla avisa al usuario ───────────────────────
try:
    from src.utils.models import MODELS
    _MODELS_OK = True
except Exception as _import_err:
    _MODELS_OK = False
    _MODELS_ERR = str(_import_err)


# ── ANSI colours ─────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

TARGET_MODELS = ["vim_tiny", "vig_tiny", "vit_tiny", "res9"]


# ── Core logic (mirrors FedLateGramServer._detect_late_layers) ────────────────

def detect_late_layers(model: nn.Module, late_fraction: float) -> list[str]:
    """Exact replica of FedLateGramServer._detect_late_layers()."""
    base_modules = [name for name, _ in model.base.named_modules() if name]
    cutoff = int(len(base_modules) * (1.0 - late_fraction))
    late_base = [f"base.{n}" for n in base_modules[cutoff:]]
    return late_base + ["classifier"]


def is_late(param_name: str, late_layer_names: list[str]) -> bool:
    return any(param_name.startswith(l) for l in late_layer_names)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def lbl(flag: bool) -> str:
    return f"{GREEN}LATE {RESET}" if flag else f"{YELLOW}EARLY{RESET}"


# ── Report for one model ──────────────────────────────────────────────────────

def report(model_name: str, dataset: str, late_fraction: float,
           pretrained: bool, show_params: bool) -> None:

    if not _MODELS_OK:
        print(f"\n{BOLD}ERROR:{RESET} Could not import MODELS registry.")
        print(f"  Root cause: {_MODELS_ERR}")
        print("  Make sure you are running from the FL-bench root with the "
              "correct virtual environment activated.")
        print("  Example:  .env/Scripts/activate  then  python scripts/inspect_late_layers.py ...\n")
        sys.exit(1)

    if model_name not in MODELS:
        print(f"\nERROR: '{model_name}' not found in MODELS. "
              f"Available: {sorted(MODELS.keys())}\n")
        sys.exit(1)

    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"  Building model: {CYAN}{BOLD}{model_name}{RESET}  (dataset={dataset}, pretrained={pretrained})")
    print(f"{'='*70}{RESET}")

    try:
        model = MODELS[model_name](dataset=dataset, pretrained=pretrained)
    except ImportError as e:
        print(f"\n  {YELLOW}SKIP{RESET} — cannot import model: {e}\n")
        return
    except Exception as e:
        print(f"\n  ERROR building model: {e}\n")
        return

    model.eval()

    late_names   = detect_late_layers(model, late_fraction)
    base_mods    = [n for n, _ in model.base.named_modules() if n]
    total_mods   = len(base_mods)
    cutoff_idx   = int(total_mods * (1.0 - late_fraction))
    n_early      = cutoff_idx
    n_late       = total_mods - cutoff_idx

    total_params = count_params(model)
    early_params = sum(
        p.numel() for n, p in model.named_parameters()
        if not is_late(n, late_names)
    )
    late_params  = total_params - early_params

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n  late_fraction  : {late_fraction:.0%}  →  cutoff at base module index {cutoff_idx}/{total_mods}")
    print(f"  {YELLOW}EARLY modules{RESET}  : {n_early}")
    print(f"  {GREEN}LATE  modules{RESET}  : {n_late} (from base) + classifier")
    print(f"  Total params   : {total_params:>12,}")
    print(f"  {YELLOW}EARLY params{RESET}   : {early_params:>12,}  ({100*early_params/total_params:.1f}%)")
    print(f"  {GREEN}LATE  params{RESET}   : {late_params:>12,}  ({100*late_params/total_params:.1f}%)")

    # ── Per-module table ──────────────────────────────────────────────────────
    print(f"\n  {'IDX':>4}  {'LABEL'}  {'MODULE PATH':<58}  {'PARAMS':>10}  {'TYPE'}")
    print(f"  {'─'*4}  {'─'*5}  {'─'*58}  {'─'*10}  {'─'*25}")

    for idx, (mod_name, module) in enumerate(model.base.named_modules()):
        if not mod_name:
            continue

        full_name  = f"base.{mod_name}"
        late_flag  = is_late(full_name, late_names)
        n_params   = count_params(module)
        mod_type   = type(module).__name__

        display = mod_name if len(mod_name) <= 58 else "…" + mod_name[-57:]
        print(
            f"  {idx:>4}  {lbl(late_flag)}  {display:<58}  {n_params:>10,}  {mod_type}"
        )

        if show_params:
            for pname, param in module.named_parameters(recurse=False):
                full_p = f"base.{mod_name}.{pname}"
                print(
                    f"  {' ':>4}  {DIM}{lbl(is_late(full_p, late_names))}"
                    f"   └─ {full_p:<55}  {param.numel():>10,}{RESET}"
                )

    # classifier — always LATE
    clf_params = count_params(model.classifier)
    clf_type   = type(model.classifier).__name__
    print(f"  {'N/A':>4}  {lbl(True)}  {'classifier':<58}  {clf_params:>10,}  {clf_type}")

    # ── Late layer names list (what the server sends to clients) ──────────────
    print(f"\n  {BOLD}late_layer_names{RESET} sent to clients ({GREEN}{len(late_names)}{RESET} entries):")
    for n in late_names:
        print(f"    {GREEN}+{RESET} {n}")

    # ── Parameter-level split (if requested) ─────────────────────────────────
    if show_params:
        print(f"\n  {BOLD}All parameters — EARLY / LATE:{RESET}")
        for pname, param in model.named_parameters():
            print(
                f"    {lbl(is_late(pname, late_names))}  {pname:<62}  {param.numel():>10,}"
            )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Inspect EARLY / LATE layer split for FedLateGram",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model",
        help="Model key from MODELS registry (e.g. res9, vit_tiny, vig_tiny, vim_tiny)",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help=f"Analyse all target models: {TARGET_MODELS}",
    )
    parser.add_argument("--dataset",        default="cifar10",
                        help="Dataset used for num_classes (default: cifar10)")
    parser.add_argument("--late_fraction",  type=float, default=0.4,
                        help="Fraction of base sub-modules treated as LATE (default: 0.4)")
    parser.add_argument("--pretrained",     action="store_true", default=False,
                        help="Load pretrained weights (default: False)")
    parser.add_argument("--show_params",    action="store_true", default=False,
                        help="Print individual parameter names within each module")
    args = parser.parse_args()

    models_to_run = TARGET_MODELS if args.all else [args.model]

    for m in models_to_run:
        report(
            model_name    = m,
            dataset       = args.dataset,
            late_fraction = args.late_fraction,
            pretrained    = args.pretrained,
            show_params   = args.show_params,
        )

    print()


if __name__ == "__main__":
    main()
