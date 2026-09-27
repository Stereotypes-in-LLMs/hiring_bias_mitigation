"""Exports the SFT training curves from Weights & Biases into a CSV the figures can read.

Only finished runs are exported, one per training config (the latest, if a config was
restarted). The checkpoint that was kept -- the one audited -- is read from the local trainer
state, so the figure can mark it on the curve.

    python scripts/export_wandb_curves.py            # -> reports/training_curves.csv
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from hiring_bias_mitigation.utils.config import REPO_ROOT, resolve_output_path  # noqa: E402
from hiring_bias_mitigation.utils.logging import get_logger  # noqa: E402

log = get_logger("export_wandb_curves")

METRICS = ("train/loss", "eval/loss", "eval/mean_token_accuracy", "eval/entropy")
#: W&B run name prefix -> the paper's main SFT runs
PREFIX = "sft-"
SKIP = ("_v2",)  # probes are not paper results


def kept_step(run_name: str) -> int | None:
    """The step of the checkpoint the trainer kept as best (the one merged and audited)."""
    out = Path(resolve_output_path(f"outputs/sft/{run_name.removeprefix(PREFIX)}"))
    states = sorted(glob.glob(str(out / "checkpoint-*" / "trainer_state.json")))
    if not states:
        return None
    best = json.loads(Path(states[-1]).read_text()).get("best_model_checkpoint") or ""
    return int(best.rsplit("-", 1)[-1]) if "checkpoint-" in best else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(REPO_ROOT / "reports" / "training_curves.csv"))
    args = parser.parse_args()

    import wandb

    path = f"{os.environ['WANDB_ENTITY'].strip()}/{os.environ['WANDB_PROJECT'].strip()}"
    runs = [r for r in wandb.Api().runs(path)
            if r.state == "finished" and r.name.startswith(PREFIX)
            and not any(s in r.name for s in SKIP)]
    latest: dict[str, object] = {}
    for r in sorted(runs, key=lambda r: r.created_at):
        latest[r.name] = r

    rows = []
    for name, r in sorted(latest.items()):
        history = r.scan_history(keys=None)
        best = kept_step(name)
        for entry in history:
            step = entry.get("train/global_step")
            for metric in METRICS:
                value = entry.get(metric)
                if step is not None and value is not None:
                    rows.append({"run": name.removeprefix(PREFIX), "step": int(step),
                                 "metric": metric, "value": float(value), "kept_step": best,
                                 "wandb_url": r.url})
        log.info("%s: kept checkpoint %s", name, best)
    frame = pd.DataFrame(rows).drop_duplicates(["run", "step", "metric"])
    frame.to_csv(args.out, index=False)
    log.info("wrote %s (%d rows, %d runs)", args.out, len(frame), frame["run"].nunique())


if __name__ == "__main__":
    main()
