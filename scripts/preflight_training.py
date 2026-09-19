"""Constructs every enabled training config's real trainer arguments, without training.

Why this exists. A signature mismatch in TrainingArguments or a TRL config is not a one-run
failure -- every config in the stage hits the same line, so the whole stage dies. It happened
twice here in one night: `warmup_ratio` (removed in transformers 5.x) took out all sixteen
training runs in ninety seconds, and the sixteen audits that depended on their checkpoints then
spent six hours loading base models to evaluate adapters that were never written. A second
pass found `max_prompt_length` gone from DPOConfig and ORPOConfig removed from TRL outright.

Building the objects costs about a second and needs no GPU. Run it before any training stage.

    python scripts/preflight_training.py
    python scripts/preflight_training.py --quiet   # exit status only
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from hiring_bias_mitigation.mitigation import training_common as C  # noqa: E402
from hiring_bias_mitigation.utils.logging import get_logger  # noqa: E402

log = get_logger("preflight_training")

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNERS = ("run_all_sft.sh", "run_all_dpo.sh")


def enabled_training_configs() -> list[Path]:
    configs = []
    for runner in RUNNERS:
        for line in (REPO_ROOT / "scripts" / runner).read_text(encoding="utf-8").splitlines():
            match = re.match(r"^  (configs/\S+\.yaml)", line)
            if match:
                configs.append(REPO_ROOT / match.group(1))
    return configs


def check(config: Path) -> None:
    """Builds the arguments this config will actually be trained with. Raises on a mismatch."""
    import trl

    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    base = C.training_arguments(
        cfg, Path("/tmp/hbm-preflight"), has_eval=True, n_train=1000
    ).to_dict()

    if cfg.get("stage") == "sft":
        trl.SFTConfig(
            **base,
            **C.supported_kwargs(
                trl.SFTConfig,
                {"max_length": cfg.get("max_seq_len", 2048), "packing": False},
                "SFTConfig",
            ),
        )
        return

    objective = cfg.get("objective", "dpo")
    cls = {"dpo": trl.DPOConfig, "kto": trl.KTOConfig}.get(objective)
    if cls is None:
        raise ValueError(f"unknown objective {objective!r} in {config.name}")
    cls(
        **base,
        **C.supported_kwargs(
            cls,
            {
                "beta": float(cfg.get("beta", 0.1)),
                "max_length": cfg.get("max_seq_len", 2048),
                "max_prompt_length": cfg.get("max_prompt_len", 1792),
            },
            cls.__name__,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quiet", action="store_true", help="suppress the per-config log")
    parser.add_argument(
        "--config", action="append", default=[],
        help="check these configs instead of whatever the runners have enabled. The smoke run "
             "uses the first of them — smoking a *different* config than the one about to "
             "train is how a custom-loss incompatibility reached the GPU unnoticed.",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="additionally run one real training step on the cheapest config. Costs minutes "
             "and exercises what a static check cannot: trainer construction, tokenisation, "
             "forward, backward and save.",
    )
    args = parser.parse_args()
    if args.quiet:
        logging.disable(logging.INFO)

    configs = [Path(c) for c in args.config] or enabled_training_configs()
    if not configs:
        log.warning("no training configs enabled in %s", ", ".join(RUNNERS))
        return

    failures = []
    for config in configs:
        try:
            check(config)
        except Exception as exc:
            failures.append((config.name, f"{type(exc).__name__}: {exc}"))

    for name, error in failures:
        log.error("%s: %s", name, error)
    if failures:
        raise SystemExit(
            f"{len(failures)} of {len(configs)} training config(s) would fail at construction. "
            "Fix them before the stage burns GPU time."
        )
    log.info("preflight ok: %d training config(s) construct cleanly", len(configs))

    if args.smoke:
        smoke(configs, explicit=bool(args.config))


def smoke(configs: list[Path], explicit: bool = False) -> None:
    """One real training step, on the cheapest config and a handful of rows.

    Constructing the config objects is not enough. TRL validates trainer *arguments against
    each other* inside the trainer constructor, not the config -- which is how a
    `formatting_func` incompatible with `completion_only_loss` passed a green preflight and
    then failed all six runs. Anything that only appears once tensors move needs a real step
    to find, and a real step on eight rows costs minutes rather than hours.
    """
    import copy

    import yaml

    from hiring_bias_mitigation.mitigation import sft as S

    # Explicit configs are smoked in the order given; otherwise pick the cheapest model.
    cheapest = configs[0] if explicit else min(
        configs, key=lambda c: 0 if "qwen3.5-4b" in c.name else 1
    )
    cfg = yaml.safe_load(cheapest.read_text(encoding="utf-8"))
    cfg = copy.deepcopy(cfg)
    cfg["output_dir"] = "outputs/_preflight_smoke"
    cfg["num_train_epochs"] = 1
    cfg["max_steps"] = 1
    cfg["save_strategy"] = "no"
    cfg["eval_steps"] = 1
    cfg["logging_steps"] = 1
    cfg["use_wandb"] = False
    cfg.setdefault("data", {})

    log.info("smoke: one training step from %s", cheapest.name)
    from hiring_bias_mitigation.utils.config import load_config

    full = load_config(str(cheapest))
    full.update({k: v for k, v in cfg.items() if k != "data"})
    # Longest rows, not the first: the smoke exists to catch what kills a real run.
    full["data"] = {**full.get("data", {}), "limit_rows": 8, "longest_rows": True}
    # Dispatch on the stage: smoking the SFT trainer for a DPO config would pass while the
    # DPO path -- two forwards per step, a reference model, a different data split -- went
    # untested, which is the same "check something other than what runs" failure as before.
    if full.get("stage") == "dpo":
        from hiring_bias_mitigation.mitigation import dpo as P

        P.train(full)
    else:
        S.train(full)
    log.info("smoke ok: a real training step completed")


if __name__ == "__main__":
    main()
