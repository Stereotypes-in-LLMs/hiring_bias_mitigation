"""Running one audit: benchmark in, raw generations out.

A run is one (model, mitigation, language, condition-set) cell of the experiment matrix. The
runner's whole job is to make the *only* difference between a baseline run and a mitigated
run be the mitigation block -- same pairs, same attributes, same decoding, same parsing. Any
other difference would end up attributed to the mitigation.

Where each mitigation family intervenes:

    prompt      -> `strategy`, in prompt construction
    scrub       -> the `cv` column, before prompt construction; under scrubbing the explicit
                   labelled field is also dropped, since a scrubber that leaves the audit's
                   own injected field in place is not scrubbing anything
    embedding   -> a forward hook on the backend's model
    sft / dpo   -> adapter merged into the weights (vLLM) or `lora_path` (transformers)

Raw generations are written to parquet before any scoring, so re-scoring a run (a new
similarity model, a corrected parser, an added measure) never costs a second inference pass.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from ..data import benchmark as B
from ..data import prompts as P
from ..data.injection import build_eval_set
from ..utils.config import resolve_output_path, sanitize
from ..utils.logging import get_logger
from ..utils.seed import set_seed
from . import backends as BK
from .parsing import parse_frame

log = get_logger(__name__)


def build_run_name(cfg: dict) -> str:
    """A stable, filename-safe identity for a run. Also the key in the results index."""
    mitigation = cfg.get("mitigation", {}) or {}
    family = mitigation.get("family", "none")
    variant = (
        mitigation.get("strategy")
        or mitigation.get("method")
        or mitigation.get("mode")
        or ("adapter" if mitigation.get("lora_path") else "")
    )
    parts = [
        sanitize(cfg["model"].split("/")[-1]),
        cfg["lang"],
        family if family != "none" else "baseline",
    ]
    if variant:
        parts.append(sanitize(variant))
    if cfg.get("run_suffix"):
        parts.append(sanitize(cfg["run_suffix"]))
    return "--".join(parts)


def build_frame(cfg: dict) -> pd.DataFrame:
    """The evaluation grid for a run, before any mitigation touches it."""
    lang = cfg["lang"]
    pairs = B.load_benchmark(lang)
    limit = cfg.get("limit_pairs")
    if limit:
        # Deterministic head, not a sample: two runs compared against each other must cover
        # the same pairs, and a seeded sample would still diverge if `limit_pairs` changed.
        pairs = pairs.head(int(limit))
    return build_eval_set(
        pairs,
        protected_groups=cfg["protected_groups"],
        lang=lang,
        conditions=tuple(cfg.get("conditions", ("explicit", "implicit", "attr_free"))),
        max_intersection_cells=cfg.get("max_intersection_cells"),
    )


def _require_adapter(lora_path: str) -> str:
    """Resolves an adapter directory, refusing before anything expensive happens.

    Checked here rather than left to the backend because the backend finds out too late: vLLM
    loads the entire base model first and only then reads the adapter, so a missing checkpoint
    costs minutes per run instead of milliseconds. When a training stage fails, every audit
    that depends on it has a missing checkpoint -- sixteen of them here, which turned a
    ninety-second training failure into six hours of loading models to evaluate weights that
    were never written.
    """
    resolved = Path(resolve_output_path(lora_path))
    if not (resolved / "adapter_config.json").is_file():
        detail = "directory does not exist" if not resolved.exists() else "no adapter_config.json"
        raise FileNotFoundError(
            f"adapter {resolved} is not usable ({detail}). Run the matching training config "
            "in configs/mitigation/{sft,dpo}/ first; its audit cannot run before it."
        )
    return str(resolved)


def _materialise_implicit(df: pd.DataFrame) -> pd.DataFrame:
    """Folds the implicit injection into the CV text itself.

    Needed before scrubbing. `build_profile` normally prepends the injected sentence at
    prompt time, which would put the attribute back *after* the scrubber ran -- the scrubbed
    run would then look identical to the baseline and the mitigation would appear to do
    nothing. Folding it in first makes the implicit condition scrubbable, which is the
    realistic case anyway: in a real CV that sentence is text the candidate wrote.
    """
    out = df.copy()
    implicit = out["condition"] == "implicit"
    out.loc[implicit, "cv"] = (
        out.loc[implicit, "implicit_injection"].astype(str) + "\n" + out.loc[implicit, "cv"]
    )
    out.loc[implicit, "implicit_injection"] = ""
    return out


def _apply_mitigation_to_frame(df: pd.DataFrame, cfg: dict, backend) -> tuple[pd.DataFrame, dict]:
    """Frame-level mitigations (currently: scrubbing). Returns (frame, metadata).

    `condition` is deliberately left alone -- the report compares a scrubbed run against the
    baseline run *within* the same condition, so renaming the condition here would put the
    two on different rows and make the comparison impossible. What the scrubbed CV changes is
    recorded in `profile_rendered` and in the run metadata instead.
    """
    out = df.assign(profile_rendered="with_attribute")
    mitigation = cfg.get("mitigation", {}) or {}
    if mitigation.get("family") != "scrub":
        return out, {}

    from ..mitigation.scrub import apply_scrub

    out = _materialise_implicit(out)
    rewriter = backend if mitigation.get("mode") in ("llm", "both") else None
    out = apply_scrub(out, mitigation, backend=rewriter)
    out["profile_rendered"] = "scrubbed"

    changed = out.get("scrub_removals", pd.Series(dtype=str)).astype(bool)
    meta = {
        "scrub_rows_changed": int(changed.sum()) if len(changed) else 0,
        "scrub_rows_total": len(out),
        "scrub_change_rate": float(changed.mean()) if len(changed) else 0.0,
    }
    return out, meta


def _build_prompts(df: pd.DataFrame, strategy: str) -> list[str]:
    """Renders the user turn for every row.

    A scrubbed row is rendered as a bare profile: the explicit labelled field is the harness's
    own injection, so leaving it in front of a scrubbed CV would hand the model the attribute
    the scrubber just removed, and the run would measure nothing.
    """
    records = df.to_dict("records")
    for row in records:
        if row.get("profile_rendered") == "scrubbed":
            row["condition"] = "attr_free"
            row["implicit_injection"] = ""
    return [P.build_prompt(row, strategy) for row in records]


def _second_pass(df: pd.DataFrame, first: pd.DataFrame, backend) -> list[str]:
    """The verifier's second call, given the first call's parsed decision and feedback."""
    prompts = [
        P.build_second_pass_prompt(row, decision, feedback)
        for row, decision, feedback in zip(
            df.to_dict("records"), first["decision"], first["feedback"].fillna("")
        )
    ]
    return backend.generate(prompts)



def _generate_resumable(backend, prompts: list[str], run_name: str, chunk: int = 4000,
                        tag: str = "main") -> list[str]:
    """Generates in chunks, keeping finished ones on disk so a crash costs one chunk.

    An audit is two hours of generation on the larger models. Losing a machine to a power cut
    halfway through used to mean starting again; now each chunk is written as it lands and a
    rerun picks up from the first unfinished one. The cache is keyed by a hash of the prompts
    themselves, so a changed config can never resume onto someone else's generations.
    """
    import hashlib

    digest = hashlib.sha256("\x00".join(prompts).encode()).hexdigest()[:16]
    cache = Path(resolve_output_path("outputs/partial")) / f"{run_name}--{tag}.parquet"
    done: list[str] = []
    if cache.is_file():
        frame = pd.read_parquet(cache)
        if len(frame) and frame["digest"].iloc[0] == digest and len(frame) <= len(prompts):
            done = frame["raw_output"].tolist()
            log.info("resuming %s: %d of %d prompts already generated", cache.name, len(done),
                     len(prompts))
        else:
            log.warning("%s is for different prompts; ignoring it", cache.name)

    cache.parent.mkdir(parents=True, exist_ok=True)
    while len(done) < len(prompts):
        batch = prompts[len(done):len(done) + chunk]
        done.extend(backend.generate(batch))
        pd.DataFrame({"raw_output": done, "digest": digest}).to_parquet(cache, index=False)
        log.info("generated %d / %d", len(done), len(prompts))
    return done[:len(prompts)]


def run_audit(cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Generates every response for one run. Returns (parsed frame, run metadata)."""
    set_seed(cfg.get("seed", 42))
    mitigation = cfg.get("mitigation", {}) or {}
    family = mitigation.get("family", "none")
    strategy = mitigation.get("strategy", "baseline") if family == "prompt" else "baseline"

    if family == "prompt":
        from ..mitigation.prompt import validate

        validate(strategy)

    activation_editor = None
    if family == "embedding":
        from ..mitigation.embedding import ActivationEditor, Eraser

        eraser_path = resolve_output_path(mitigation["eraser_path"])
        if not Path(eraser_path).exists():
            raise FileNotFoundError(
                f"no fitted eraser at {eraser_path}. Run scripts/fit_eraser.py with this "
                "config first -- the eraser must be fitted on the training pool, never on "
                "the benchmark."
            )
        activation_editor = ActivationEditor(
            eraser=Eraser.load(eraser_path), strength=float(mitigation.get("strength", 1.0))
        )

    backend_cfg = dict(cfg)
    if family in ("sft", "dpo") and mitigation.get("lora_path"):
        adapter = _require_adapter(mitigation["lora_path"])
        if backend_cfg.get("backend", "vllm") == "vllm":
            # vLLM's LoRA path does not reproduce Qwen3.5 adapters; serve merged weights.
            # `cfg["model"]` is untouched, so the run name still names the base model.
            from hiring_bias_mitigation.mitigation.merge import merge

            backend_cfg["model"] = str(merge(adapter))
        else:
            backend_cfg["lora_path"] = adapter

    backend = BK.build_backend(backend_cfg, activation_editor=activation_editor)
    started = time.time()

    try:
        df = build_frame(cfg)
        df, mitigation_meta = _apply_mitigation_to_frame(df, cfg, backend)
        log.info("run %s: %d prompts", build_run_name(cfg), len(df))

        raw = _generate_resumable(backend, _build_prompts(df, strategy),
                                  build_run_name(cfg))
        df = df.assign(raw_output=raw)
        parsed = parse_frame(df)

        if family == "prompt" and strategy == "second_pass_verification":
            log.info("second verification pass over %d items", len(parsed))
            raw2 = _second_pass(df, parsed, backend)
            # Keep the first pass alongside the revised answer: the interesting quantity is
            # how often the verifier *changed* a decision, and on which attributes.
            first_decision = parsed["decision"]
            parsed = parse_frame(df.assign(raw_output=raw2))
            parsed["first_pass_decision"] = first_decision.values
            parsed["verifier_changed_decision"] = (
                parsed["decision"] != first_decision.values
            ).astype(float)
            mitigation_meta["verifier_changed_rate"] = float(
                parsed["verifier_changed_decision"].mean()
            )
    finally:
        backend.close()

    generation = BK.GenerationConfig(**cfg.get("generation", {})).as_dict()
    meta = {
        "run_name": build_run_name(cfg),
        "model": cfg["model"],
        # What the backend actually loaded: a merged checkpoint for vLLM-served adapters.
        "served_model": backend_cfg["model"],
        "lang": cfg["lang"],
        "backend": cfg.get("backend", "vllm"),
        "protected_groups": cfg["protected_groups"],
        "conditions": list(cfg.get("conditions", ("explicit", "implicit", "attr_free"))),
        "mitigation": {**mitigation, **mitigation_meta},
        "prompt_strategy": strategy,
        "generation": generation,
        "seed": cfg.get("seed", 42),
        "limit_pairs": cfg.get("limit_pairs"),
        "n_prompts": len(parsed),
        "wall_seconds": round(time.time() - started, 1),
        "config_path": cfg.get("_config_path"),
    }
    return parsed, meta


def _drop_partial(run_name: str) -> None:
    """The finished artifact supersedes the chunk cache."""
    folder = Path(resolve_output_path("outputs/partial"))
    for path in folder.glob(f"{run_name}--*.parquet"):
        path.unlink(missing_ok=True)


def save_raw(parsed: pd.DataFrame, meta: dict, output_dir: str | Path) -> Path:
    """Persists generations plus metadata so scoring never needs the GPU again."""
    output_dir = Path(resolve_output_path(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{meta['run_name']}.parquet"
    parsed.to_parquet(path, index=False)
    (output_dir / f"{meta['run_name']}.meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("wrote %d rows to %s", len(parsed), path)
    _drop_partial(meta["run_name"])
    return path


def load_raw(path: str | Path) -> tuple[pd.DataFrame, dict]:
    path = Path(path)
    # Not `with_suffix`: run names carry dots ("Qwen3.5-4B--en--baseline"), and with_suffix
    # treats everything after the first dot as an extension, so it would look for
    # "Qwen3.meta.json". `stem` strips only the real ".parquet".
    meta_path = path.parent / f"{path.stem}.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return pd.read_parquet(path), meta
