"""Folds a LoRA adapter into its base model, for serving without LoRA.

vLLM's LoRA path does not reproduce Qwen3.5 adapters faithfully: on the same benchmark prompts
the base model's decisions agree 98.9% between HuggingFace and vLLM, an SFT adapter's only
82.5%. The adapter changed 17.5% of decisions under HF + PEFT, the framework it was trained
in, and 0.3% under the vLLM audit. Audits therefore serve adapters as merged checkpoints,
which vLLM loads with no LoRA code involved (`eval/runner.py`).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from hiring_bias_mitigation.utils.config import resolve_output_path
from hiring_bias_mitigation.utils.logging import get_logger

log = get_logger(__name__)


def merged_dir_for(adapter: str | Path) -> Path:
    """`sft/x` -> `merged/x`; `sft/x/checkpoint-250` -> `merged/x--checkpoint-250`."""
    adapter = Path(adapter)
    name = adapter.name
    if name.startswith("checkpoint-"):
        name = f"{adapter.parent.name}--{name}"
    return Path(resolve_output_path("outputs/merged")) / name


def conform_to_base(out: Path, base_name: str, base_dir: Path) -> None:
    """Give the merged checkpoint the base's exact layout.

    Training loads the text model only (`Qwen3_5ForCausalLM`), so the save lacks the base's
    vision tower and carries a text-only config. vLLM would then serve it through a different
    model class than the baseline audit. Copying the base config and the untouched tensors the
    save lacks makes the merged checkpoint differ from the base in the merged weights only.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    base_map = json.loads((base_dir / "model.safetensors.index.json").read_text())["weight_map"]
    ours: dict[str, str] = {}
    for f in sorted(out.glob("*.safetensors")):
        if f.is_symlink():
            raise SystemExit(f"{f} is a link into the HF cache, not a merged tensor file")
        with safe_open(f, "pt") as fh:
            ours.update(dict.fromkeys(fh.keys(), f.name))
    unknown = set(ours) - set(base_map)
    if unknown:
        raise SystemExit(f"merged tensors the base does not have: {sorted(unknown)[:5]}")
    missing = sorted(set(base_map) - set(ours))
    if missing:
        from huggingface_hub import snapshot_download

        shards = sorted({base_map[k] for k in missing})
        base_w = Path(snapshot_download(base_name, allow_patterns=shards))
        extra = {}
        for shard in shards:
            with safe_open(base_w / shard, "pt") as fh:
                extra.update({k: fh.get_tensor(k) for k in missing if base_map[k] == shard})
        save_file(extra, out / "base-untouched.safetensors", metadata={"format": "pt"})
        ours.update(dict.fromkeys(extra, "base-untouched.safetensors"))
    total = sum((out / f).stat().st_size for f in set(ours.values()))
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": dict(sorted(ours.items()))}, indent=2))
    shutil.copy2(base_dir / "config.json", out / "config.json")
    log.info("conformed to base layout: %d merged + %d base tensors", len(ours) - len(missing),
             len(missing))


def merge(adapter: str | Path, force: bool = False) -> Path:
    """Merges `adapter` into its base once and returns the merged directory."""
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter = Path(resolve_output_path(str(adapter)))
    out = merged_dir_for(adapter)
    marker = out / "merged_from.json"
    if marker.exists() and not force:
        recorded = json.loads(marker.read_text())
        # Retraining writes a new adapter to the same path; a stale merge must not be reused.
        if recorded.get("adapter_mtime") == _adapter_mtime(adapter):
            log.info("%s already merged", out)
            return out
        log.info("adapter %s changed since it was merged; re-merging", adapter)

    base_name = json.loads((adapter / "adapter_config.json").read_text())["base_model_name_or_path"]
    log.info("merging %s into %s", adapter, base_name)
    model = AutoModelForCausalLM.from_pretrained(base_name, dtype=torch.bfloat16,
                                                 device_map="cpu", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    if out.exists():
        shutil.rmtree(out)  # never mix shards from an earlier, interrupted merge
    out.mkdir(parents=True)
    model.save_pretrained(out, safe_serialization=True)
    del model
    AutoTokenizer.from_pretrained(base_name, trust_remote_code=True).save_pretrained(out)
    # Keep whatever else the base ships for loading (chat template, generation config,
    # preprocessor files a multimodal checkpoint declares). The snapshot dir also holds any
    # base weights downloaded earlier; copy only the small loading files, never tensors or the
    # base's weight index.
    base_dir = Path(snapshot_download(base_name, allow_patterns=["*.json", "*.jinja", "*.txt"]))
    for f in base_dir.iterdir():
        if (f.suffix in {".json", ".jinja", ".txt"} and f.name != "model.safetensors.index.json"
                and not (out / f.name).exists()):
            shutil.copy2(f, out / f.name)
    conform_to_base(out, base_name, base_dir)
    marker.write_text(json.dumps({"base_model": base_name, "adapter": str(adapter),
                                  "adapter_mtime": _adapter_mtime(adapter)}, indent=2))
    log.info("wrote %s", out)
    return out


def _adapter_mtime(adapter: Path) -> float:
    return max(f.stat().st_mtime for f in adapter.glob("adapter_model*"))
