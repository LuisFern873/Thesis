"""
Static layer analysis for res9, vit_tiny, vig_tiny, vim_tiny.
Reconstructs each model's base structure using only pure PyTorch (no torchvision/timm)
so it works even with a broken venv. Mirrors exactly the _detect_late_layers() logic.
Run:  python scripts/_static_layer_analysis.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn

GREEN  = "\033[92m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
CYAN   = "\033[96m"
DIM    = "\033[2m"


def lbl(flag):
    return f"{GREEN}LATE {RESET}" if flag else f"{YELLOW}EARLY{RESET}"


def analyse(model_name, base: nn.Module, classifier: nn.Module, late_fraction=0.4):
    mods = [n for n, _ in base.named_modules() if n]
    total = len(mods)
    cutoff = int(total * (1.0 - late_fraction))
    late_names = [f"base.{n}" for n in mods[cutoff:]] + ["classifier"]

    def is_late(pn):
        return any(pn.startswith(l) for l in late_names)

    base_params = {f"base.{n}": p.numel() for n, p in base.named_parameters()}
    clf_params  = {f"classifier.{n}": p.numel() for n, p in classifier.named_parameters()}
    all_p       = {**base_params, **clf_params}

    total_params = sum(all_p.values())
    early_params = sum(v for k, v in all_p.items() if not is_late(k))
    late_params  = total_params - early_params

    print(f"\n{BOLD}{'='*72}{RESET}")
    print(f"  {CYAN}{BOLD}{model_name}{RESET}")
    print(f"{'='*72}")
    print(f"  Base sub-modules : {total}  |  cutoff={cutoff}  |  late_fraction={late_fraction:.0%}")
    print(f"  {YELLOW}EARLY modules{RESET}: {cutoff}   {GREEN}LATE modules{RESET}: {total-cutoff} (from base) + classifier")
    print(f"  Total params : {total_params:>12,}")
    print(f"  {YELLOW}EARLY params{RESET}: {early_params:>12,}  ({100*early_params/total_params:.1f}%)")
    print(f"  {GREEN}LATE  params{RESET}: {late_params:>12,}  ({100*late_params/total_params:.1f}%)")

    W_idx  = 4
    W_lbl  = 5
    W_mod  = 52
    W_par  = 10
    W_type = 26

    header_idx  = "IDX".rjust(W_idx)
    header_lbl  = "LABEL".ljust(W_lbl)
    header_mod  = "MODULE PATH".ljust(W_mod)
    header_par  = "PARAMS".rjust(W_par)
    header_type = "TYPE".ljust(W_type)

    print()
    print(f"  {header_idx}  {header_lbl}  {header_mod}  {header_par}  {header_type}")
    print(f"  {'─'*W_idx}  {'─'*W_lbl}  {'─'*W_mod}  {'─'*W_par}  {'─'*W_type}")

    for idx, (n, m) in enumerate(base.named_modules()):
        if not n:
            continue
        flag   = is_late(f"base.{n}")
        n_par  = sum(p.numel() for p in m.parameters())
        typ    = type(m).__name__
        disp   = n if len(n) <= W_mod else "..." + n[-(W_mod-3):]
        print(f"  {str(idx).rjust(W_idx)}  {lbl(flag)}  {disp.ljust(W_mod)}  {str(n_par).rjust(W_par)}  {typ}")

    cp = sum(p.numel() for p in classifier.parameters())
    ct = type(classifier).__name__
    print(f"  {'N/A'.rjust(W_idx)}  {lbl(True)}  {'classifier'.ljust(W_mod)}  {str(cp).rjust(W_par)}  {ct}")

    print(f"\n  {BOLD}late_layer_names{RESET} ({GREEN}{len(late_names)}{RESET} entries) — sent to every client:")
    for n in late_names:
        print(f"    {GREEN}+{RESET} {n}")


# ─────────────────────────────────────────────────────────────────────────────
# res9  — pure-PyTorch reconstruction
# ─────────────────────────────────────────────────────────────────────────────
class _ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return x + self.block(x)


def make_res9(num_classes=10):
    base = nn.Sequential(
        nn.Conv2d(3, 64,  3, padding=1, bias=False), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        _ResBlock(128),
        nn.Conv2d(128, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        nn.Conv2d(256, 512, 3, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        _ResBlock(512),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )
    clf = nn.Linear(512, num_classes)
    return base, clf


# ─────────────────────────────────────────────────────────────────────────────
# vit_tiny  — structural reconstruction (no timm needed)
# ViT-Tiny patch16/224: embed_dim=192, depth=12, num_heads=3
# named_modules from timm's VisionTransformer:
#   patch_embed, patch_embed.proj
#   pos_drop
#   blocks, blocks.0 … blocks.11
#     blocks.N.norm1, blocks.N.attn, blocks.N.attn.qkv,
#     blocks.N.attn.attn_drop, blocks.N.attn.proj, blocks.N.attn.proj_drop
#     blocks.N.ls1 (if used), blocks.N.drop_path1
#     blocks.N.norm2, blocks.N.mlp, blocks.N.mlp.fc1, blocks.N.mlp.act,
#     blocks.N.mlp.drop1, blocks.N.mlp.fc2, blocks.N.mlp.drop2
#     blocks.N.ls2, blocks.N.drop_path2
#   norm
#   head  (replaced by Identity)
# ─────────────────────────────────────────────────────────────────────────────
def _make_vit_tiny_base_approx():
    """Approximate ViT-Tiny base module tree for layer-split analysis.
    Uses the same sub-module names timm exposes so _detect_late_layers gives
    the same result as with the real model.
    """
    D = 192   # embed_dim
    H = 3     # num_heads
    MLP = D * 4  # mlp_ratio=4

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.LayerNorm(D)
            self.attn  = nn.MultiheadAttention(D, H, batch_first=True)
            self.norm2 = nn.LayerNorm(D)
            self.mlp   = nn.Sequential(
                nn.Linear(D, MLP), nn.GELU(), nn.Linear(MLP, D)
            )
        def forward(self, x): return x

    class PatchEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Conv2d(3, D, kernel_size=16, stride=16)
        def forward(self, x): return x

    class ViTTinyBase(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = PatchEmbed()
            self.pos_drop    = nn.Dropout(0.0)
            self.blocks      = nn.Sequential(*[Block() for _ in range(12)])
            self.norm        = nn.LayerNorm(D)
            self.head        = nn.Identity()    # replaced in ViTTiny.__init__
        def forward(self, x): return x

    return ViTTinyBase()


def make_vit_tiny(num_classes=10):
    base = _make_vit_tiny_base_approx()
    clf  = nn.Linear(192, num_classes)
    return base, clf


# ─────────────────────────────────────────────────────────────────────────────
# vim_tiny  — structural reconstruction
# VisionMamba depth=12 sub-modules exposed by named_modules:
#   patch_embed, patch_embed.proj
#   pos_drop
#   layers, layers.0 … layers.11
#     layers.N.norm, layers.N.mixer (MambaBlock or BiMamba)
#     layers.N.drop_path
#   norm_f
# ─────────────────────────────────────────────────────────────────────────────
def _make_vim_tiny_base_approx():
    D = 192

    class MixerBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm  = nn.LayerNorm(D)
            self.mixer = nn.Linear(D, D)      # placeholder for SSM block
            self.drop_path = nn.Identity()
        def forward(self, x): return x

    class PatchEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Conv2d(3, D, kernel_size=16, stride=16)
        def forward(self, x): return x

    class VimTinyBase(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = PatchEmbed()
            self.pos_drop    = nn.Dropout(0.0)
            self.layers      = nn.Sequential(*[MixerBlock() for _ in range(12)])
            self.norm_f      = nn.LayerNorm(D)
        def forward(self, x): return x

    return VimTinyBase()


def make_vim_tiny(num_classes=10):
    base = _make_vim_tiny_base_approx()
    clf  = nn.Linear(192, num_classes)
    return base, clf


# ─────────────────────────────────────────────────────────────────────────────
# vig_tiny  — structural reconstruction
# _ViGBase sub-modules:
#   stem, stem.0…stem.N  (4×Conv stack)
#   pos_embed  (Parameter, not a sub-module → not in named_modules)
#   blocks, blocks.0 … blocks.11
#     blocks.N.0 (Grapher), blocks.N.1 (FFN)
# ─────────────────────────────────────────────────────────────────────────────
def _make_vig_tiny_base_approx():
    D = 192

    class Grapher(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(D, D)
            self.fc2 = nn.Linear(D, D)
        def forward(self, x): return x

    class FFN(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(D, D*4)
            self.fc2 = nn.Linear(D*4, D)
        def forward(self, x): return x

    class Stem(nn.Module):
        def __init__(self):
            super().__init__()
            # 4×(Conv-BN-ReLU) downsampling from 224→14
            self.convs = nn.Sequential(
                nn.Conv2d(3,   48, 3, stride=2, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True),
                nn.Conv2d(48,  96, 3, stride=2, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True),
                nn.Conv2d(96,  D,  3, stride=2, padding=1), nn.BatchNorm2d(D),  nn.ReLU(inplace=True),
                nn.Conv2d(D,   D,  3, stride=2, padding=1), nn.BatchNorm2d(D),  nn.ReLU(inplace=True),
            )
        def forward(self, x): return self.convs(x)

    class ViGTinyBase(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem   = Stem()
            # pos_embed is a Parameter, won't appear in named_modules
            self.blocks = nn.Sequential(*[
                nn.Sequential(Grapher(), FFN()) for _ in range(12)
            ])
        def forward(self, x): return x

    return ViGTinyBase()


def make_vig_tiny(num_classes=10):
    base = _make_vig_tiny_base_approx()
    clf  = nn.Linear(192, num_classes)
    return base, clf


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    late_fraction = 0.4
    print(f"\n{BOLD}FedLateGram — EARLY / LATE layer split  (late_fraction={late_fraction:.0%}){RESET}")
    print(f"{DIM}Criterion: last {late_fraction:.0%} of base.named_modules() + classifier{RESET}")

    analyse("res9",     *make_res9(),     late_fraction)
    analyse("vit_tiny", *make_vit_tiny(), late_fraction)
    analyse("vim_tiny", *make_vim_tiny(), late_fraction)
    analyse("vig_tiny", *make_vig_tiny(), late_fraction)

    print()
