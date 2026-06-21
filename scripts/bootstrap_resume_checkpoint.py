"""bootstrap_resume_checkpoint.py — Build a training_state.pt from legacy CKA checkpoints.

For runs that were interrupted BEFORE the resume feature existed, only the
per-round CKA checkpoints (``round_XXXX/global.pt``) are available.
This script synthesises a ``training_state.pt`` from the last available
``global.pt`` so that training can be resumed with:

    common.resume_checkpoint=<run_dir>/checkpoints/training_state.pt

What is recovered
-----------------
- ``public_model_params`` : loaded from the last ``round_XXXX/global.pt``
- ``current_epoch``        : derived from the round directory name (0-based)
- ``client_sample_stream`` : re-generated deterministically from the original
                             seed (identical to what the server would have
                             generated, provided ``join_ratio == 1.0`` and no
                             straggler — both true for the thesis experiments)

What is NOT recovered (reset to defaults)
------------------------------------------
- ``client_optimizer_states``      → empty dicts (equivalent to a fresh start
                                     per round, which matches
                                     ``reset_optimizer_on_global_epoch: true``)
- ``client_lr_scheduler_states``   → empty dicts (same reasoning)
- ``clients_personal_model_params``→ empty dicts (FedAvg/FedProx have none)
- ``aggregated_client_metrics``    → empty lists  (learning-curve history lost,
                                     but training itself is unaffected)

Usage
-----
    python scripts/bootstrap_resume_checkpoint.py \\
        --run_dir logs/runs/brain_tumor_alpha1000.0_res9_ckadriftfedprox_seed42

    # Optionally target a specific round (defaults to the highest available):
    python scripts/bootstrap_resume_checkpoint.py \\
        --run_dir logs/runs/brain_tumor_alpha1000.0_res9_ckadriftfedprox_seed42 \\
        --round 40

After running, verify with:
    python -c "import torch; s=torch.load('...checkpoints/training_state.pt', weights_only=False); print('epoch', s['current_epoch'], '| keys', list(s['public_model_params'].keys())[:3])"
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_round_dirs(checkpoints_dir: Path) -> dict[int, Path]:
    """Return a mapping {round_idx (1-based): directory_path}."""
    pattern = re.compile(r"^round_(\d{4})$")
    rounds = {}
    for entry in checkpoints_dir.iterdir():
        m = pattern.match(entry.name)
        if m and entry.is_dir():
            idx = int(m.group(1))
            rounds[idx] = entry
    return rounds


def rebuild_client_sample_stream(
    seed: int,
    num_clients: int,
    global_epoch: int,
    join_ratio: float = 1.0,
) -> list[list[int]]:
    """Re-create the deterministic client-sampling stream.

    Mirrors exactly the logic in ``FedAvgServer.__init__``:

        fix_random_seed(seed)           # sets random.seed(seed)
        # straggler block skipped (straggler_ratio == 0 for thesis runs)
        client_sample_stream = [
            random.sample(train_clients, max(1, int(client_num * join_ratio)))
            for _ in range(global_epoch)
        ]

    ASSUMPTION: ``straggler_ratio == 0``.  If the original run used stragglers,
    the stream here will NOT match (the shuffle would advance the RNG state).
    For all thesis experiments this assumption holds.

    ``train_clients`` is ``list(range(num_clients))`` for all thesis datasets
    (the partition always puts every client in the train split).
    """
    random.seed(seed)   # replicate fix_random_seed(seed)
    train_clients = list(range(num_clients))
    k = max(1, int(num_clients * join_ratio))
    stream = [random.sample(train_clients, k) for _ in range(global_epoch)]
    return stream


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesise training_state.pt from legacy CKA checkpoints."
    )
    parser.add_argument(
        "--run_dir",
        required=True,
        help="Path to the interrupted run directory "
             "(e.g. logs/runs/brain_tumor_alpha1000.0_res9_ckadriftfedprox_seed42)",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=None,
        help="1-based round index to resume FROM (defaults to the highest available).",
    )
    parser.add_argument(
        "--join_ratio",
        type=float,
        default=1.0,
        help="Client join ratio used in the original run (default: 1.0).",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    checkpoints_dir = run_dir / "checkpoints"

    if not checkpoints_dir.is_dir():
        sys.exit(f"ERROR: checkpoints directory not found: {checkpoints_dir}")

    # ---- load run_metadata.json ----------------------------------------
    metadata_path = checkpoints_dir / "run_metadata.json"
    if not metadata_path.is_file():
        sys.exit(f"ERROR: run_metadata.json not found in {checkpoints_dir}")

    with open(metadata_path) as f:
        meta = json.load(f)

    seed: int = meta["seed"]
    num_clients: int = meta["num_clients"]
    global_epoch: int = meta["global_epoch"]

    print(f"Run metadata: seed={seed}, clients={num_clients}, global_epoch={global_epoch}")

    # ---- find available rounds -----------------------------------------
    round_dirs = find_round_dirs(checkpoints_dir)
    if not round_dirs:
        sys.exit("ERROR: No round_XXXX directories found in checkpoints/")

    available = sorted(round_dirs.keys())
    print(f"Available rounds: {available}")

    target_round = args.round if args.round is not None else max(available)
    if target_round not in round_dirs:
        sys.exit(
            f"ERROR: Round {target_round} not found. Available: {available}"
        )

    round_dir = round_dirs[target_round]
    global_pt = round_dir / "global.pt"
    if not global_pt.is_file():
        sys.exit(f"ERROR: global.pt not found in {round_dir}")

    print(f"Using global weights from: {global_pt}")

    # ---- load global model params -------------------------------------
    public_model_params = torch.load(global_pt, map_location="cpu", weights_only=True)
    print(f"Loaded {len(public_model_params)} parameter tensors.")

    # ---- rebuild client sample stream ---------------------------------
    # NOTE: The FedAvgServer __init__ seeds the global Python RNG with
    # fix_random_seed(seed), then immediately draws client_sample_stream.
    # We replicate that exact call sequence here.
    client_sample_stream = rebuild_client_sample_stream(
        seed=seed,
        num_clients=num_clients,
        global_epoch=global_epoch,
        join_ratio=args.join_ratio,
    )
    print(
        f"Rebuilt client_sample_stream for {global_epoch} rounds "
        f"(first 3: {client_sample_stream[:3]})"
    )

    # ---- assemble state dict ------------------------------------------
    # current_epoch is 0-based index of the LAST COMPLETED round.
    # round_XXXX directories use 1-based naming, so subtract 1.
    current_epoch = target_round - 1

    state = {
        "current_epoch": current_epoch,
        "public_model_params": public_model_params,
        # These cannot be recovered; defaults are safe because
        # reset_optimizer_on_global_epoch=true (optimizer reset each round anyway)
        "client_optimizer_states": {i: {} for i in range(num_clients)},
        "client_lr_scheduler_states": {i: {} for i in range(num_clients)},
        "clients_personal_model_params": {i: {} for i in range(num_clients)},
        # Metric history is lost; training itself is unaffected
        "aggregated_client_metrics": {
            "before": {"train": [], "val": [], "test": []},
            "after": {"train": [], "val": [], "test": []},
        },
        "client_sample_stream": client_sample_stream,
    }

    # ---- write atomically --------------------------------------------
    dest = checkpoints_dir / "training_state.pt"
    tmp = checkpoints_dir / "training_state.pt.tmp"
    torch.save(state, tmp)
    os.replace(tmp, dest)

    print(f"\n✓ Wrote {dest}")
    print(f"  Resume from round {current_epoch + 2}/{global_epoch}  "
          f"(rounds 1–{target_round} already done)")
    print(f"\nResume command:")
    print(
        f'  python main.py --config-name ckadriftfedprox method=ckadriftfedprox \\\n'
        f'    dataset.name={meta["dataset_name"]} \\\n'
        f'    dataset.partition_dir={meta["partition_dir"]} \\\n'
        f'    model.name={meta["model_name"]} \\\n'
        f'    common.seed={seed} common.global_epoch={global_epoch} \\\n'
        f'    mode=serial \\\n'
        f'    "hydra.run.dir={run_dir}" \\\n'
        f'    "common.resume_checkpoint={dest}"'
    )


if __name__ == "__main__":
    main()
