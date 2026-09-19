"""Fits the concept-erasure operator for the embedding-based mitigation.

Activations are collected from the **training pool**, never from the benchmark. Fitting an
eraser on the evaluation set and then reporting the evaluation numbers would be fitting to
the test set with extra steps, and the resulting improvement would be meaningless. The
generated training data already carries the counterfactual structure this needs -- the same
CV under several attribute values -- which is what makes the fitted direction attribute-
specific rather than merely CV-specific.

    python scripts/fit_eraser.py --config configs/mitigation/embedding/<x>.yaml

Prints linear-probe accuracy before and after erasure. Before-accuracy near chance means
there was no linear attribute signal at these layers to remove, and any fairness change the
intervention produces needs a different explanation -- record that in the report rather than
claiming the erasure worked.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiring_bias_mitigation.data.benchmark import assert_no_leakage  # noqa: E402
from hiring_bias_mitigation.mitigation import embedding as E  # noqa: E402
from hiring_bias_mitigation.mitigation import training_common as C  # noqa: E402
from hiring_bias_mitigation.utils.config import load_config, resolve_output_path  # noqa: E402
from hiring_bias_mitigation.utils.logging import get_logger  # noqa: E402
from hiring_bias_mitigation.utils.seed import set_seed  # noqa: E402

log = get_logger("fit_eraser")


def load_fitting_rows(cfg: dict) -> pd.DataFrame:
    """The prompts to collect activations from, taken from the SFT training split."""
    data_cfg = cfg["data"]
    root = Path(resolve_output_path(data_cfg["artifacts_dir"]))
    path = root / "sft_train.parquet"
    if not path.exists():
        raise SystemExit(
            f"{path} is missing. Run scripts/generate_training_data.py first -- the eraser "
            "must be fitted on generated training data, not on the benchmark."
        )
    df = pd.read_parquet(path)
    df = C.apply_data_filters(df, data_cfg)
    assert_no_leakage(df, "fit_eraser.load_fitting_rows")

    group = cfg["mitigation"]["protected_group"]
    df = df[df["protected_group"] == group]
    if df.empty:
        raise SystemExit(f"no training rows for protected group {group!r}")

    limit = int(cfg["mitigation"].get("n_fitting_rows", 2000))
    if len(df) > limit:
        df = df.sample(n=limit, random_state=cfg.get("seed", 42))
    return df.reset_index(drop=True)


def _activations(cfg, mitigation, rows, layers, refit: bool):
    """The activation matrix and its labels, collected once per model/language/layer set.

    Collection is a forward pass over every fitting prompt and takes hours; the matrix it
    produces depends on the model, the language, the layers and the prompts -- **not** on the
    erasure method. LEACE, INLP and mean-diff for one model and language therefore share it,
    and collecting it once instead of three times is the difference between a day of GPU and
    three.

    The key covers everything the matrix depends on, so a changed fitting set or layer index
    misses the cache rather than silently reusing a matrix that no longer matches.
    """
    import hashlib
    import json

    import numpy as np

    key = json.dumps(
        {
            "model": cfg["model"],
            "lang": cfg.get("lang"),
            "layers": list(layers),
            "data_config": cfg.get("data_config"),
            "n_rows": len(rows),
            "prompt_digest": hashlib.sha256(
                "\u0000".join(rows["prompt"].tolist()).encode("utf-8")
            ).hexdigest()[:16],
            "dtype": cfg.get("dtype", "bfloat16"),
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    cache_dir = Path(resolve_output_path("outputs/erasers/_activations"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{digest}.npz"

    if cache_path.exists() and not refit:
        blob = np.load(cache_path, allow_pickle=True)
        log.info("reusing cached activations from %s", cache_path)
        return blob["activations"], blob["labels"]

    log.info(
        "collecting activations for %d prompts at layers %s from %s",
        len(rows), layers, cfg["model"],
    )
    model, tokenizer = C.load_model_and_tokenizer(
        {"base_model": cfg["model"], "dtype": cfg.get("dtype", "bfloat16")}, for_training=False
    )
    model.eval()
    activations = E.collect_activations(
        model, tokenizer, rows["prompt"].tolist(), layers,
        batch_size=int(mitigation.get("batch_size", 8)),
    )
    labels = rows["protected_attr"].to_numpy()

    np.savez_compressed(cache_path, activations=activations, labels=labels)
    (cache_dir / f"{digest}.key.json").write_text(key, encoding="utf-8")
    log.info("cached activations at %s", cache_path)
    return activations, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true", help="refit even if the operator exists")
    parser.add_argument(
        "--refit-activations", action="store_true",
        help="ignore the cached activation matrix and collect it again",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    mitigation = cfg["mitigation"]
    set_seed(cfg.get("seed", 42))

    out_path = Path(resolve_output_path(mitigation["eraser_path"]))
    if out_path.exists() and not args.force:
        log.info("%s already exists; pass --force to refit", out_path)
        return

    rows = load_fitting_rows(cfg)
    layers = tuple(mitigation.get("layers", (16,)))
    activations, labels = _activations(cfg, mitigation, rows, layers, args.refit_activations)

    before = E.linear_probe_accuracy(activations, labels, seed=cfg.get("seed", 42))
    eraser = E.fit_eraser(
        activations, labels,
        method=mitigation.get("method", "leace"),
        layers=layers,
        inlp_iterations=int(mitigation.get("inlp_iterations", 8)),
        seed=cfg.get("seed", 42),
    )
    erased = eraser.mean + (activations - eraser.mean) @ eraser.projection.T
    after = E.linear_probe_accuracy(erased, labels, seed=cfg.get("seed", 42))
    chance = float(pd.Series(labels).value_counts(normalize=True).max())

    eraser.save(out_path)
    diagnostics = {
        "eraser_path": str(out_path),
        "model": cfg["model"],
        "method": eraser.method,
        "layers": list(layers),
        "protected_group": mitigation["protected_group"],
        "n_fitting_rows": len(rows),
        "hidden_size": int(activations.shape[1]),
        "rank_removed": eraser.rank_removed,
        "probe_accuracy_before": round(before, 4),
        "probe_accuracy_after": round(after, 4),
        "majority_class_rate": round(chance, 4),
        "relative_change_l2": float(
            np.linalg.norm(erased - activations) / max(np.linalg.norm(activations), 1e-9)
        ),
    }
    out_path.with_suffix(".diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    print(json.dumps(diagnostics, indent=2))
    if before <= chance + 0.02:
        log.warning(
            "probe accuracy before erasure (%.3f) is at the majority-class rate (%.3f): there "
            "was no linear attribute signal at these layers to erase. Try other layers, or "
            "report this as a negative result about linear representation.",
            before, chance,
        )


if __name__ == "__main__":
    main()
