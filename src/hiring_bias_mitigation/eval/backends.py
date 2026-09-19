"""Generation backends. One interface, two implementations, chosen per run.

`vllm`
    Default for every audit and for training-data generation. Batched, and it loads a LoRA
    adapter alongside a base model, which is how the SFT and DPO runs are evaluated without
    materialising a merged checkpoint per experiment.

`transformers`
    Required by the embedding-based mitigation, which needs forward hooks on the residual
    stream -- vLLM's scheduler gives no stable place to hang those. Also the fallback on a
    machine without a working vLLM build. Slower; fine for a single condition, painful for a
    full sweep.

Decoding configuration is recorded on every run and printed in the report. Audit-paper
reporting requirement 5: decoding is stochastic, a single sample cannot separate bias from
sampling variance, and the inconsistency rate is partly a measure of that variance. Greedy
decoding (`temperature: 0.0`) is the default here for exactly that reason -- it removes the
variance so a change between two runs is attributable to the mitigation. Set `n_samples > 1`
with a non-zero temperature when you want to measure the variance instead of removing it.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Protocol

from ..utils.logging import get_logger

log = get_logger(__name__)

# vLLM defaults to forking its EngineCore worker. On this stack that fails for the Gemma
# architectures with "Cannot re-initialize CUDA in forked subprocess" -- something in their
# startup path touches the CUDA context before the fork, where Qwen's does not. The two Qwen
# runs of a six-run sweep therefore succeeded and all four Gemma-family runs died in seconds.
#
# `spawn` starts the worker from a clean interpreter, which sidesteps it entirely. The cost is
# that the child re-imports the parent's __main__, so every entry point must be a real file
# guarded by `if __name__ == "__main__"` -- true of every script here, but it is why a
# `python - <<EOF` heredoc fails under spawn with a confusing FileNotFoundError on '<stdin>'.
#
# setdefault, and set at import time: this module is always imported before vLLM itself
# (vLLM is imported lazily inside the backends), and an explicit VLLM_WORKER_MULTIPROC_METHOD
# in the environment still wins.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 42
    n_samples: int = 1

    def as_dict(self) -> dict:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "n_samples": self.n_samples,
            "greedy": self.temperature == 0.0,
        }


class Backend(Protocol):
    def generate(self, prompts: list[str]) -> list[str]:
        ...

    def close(self) -> None:
        ...


#: Passed into the chat template. Thinking mode is OFF by default, for two reasons.
#:
#: Practical: a reasoning model spends its whole budget on a visible thinking trace and gets
#: truncated before it ever emits the JSON. Qwen3.5-4B produced a 100% parse-failure rate at
#: max_new_tokens=512 this way -- every response was a "Thinking Process:" preamble cut off
#: mid-sentence. Raising the budget instead would multiply the cost of a ~1M-generation audit
#: several times over to pay for text no measure reads.
#:
#: Scientific: the audit study this work extends evaluated non-reasoning models, so
#: non-thinking mode is what keeps a baseline here comparable to its published baseline. It
#: also keeps the arm honest -- "does an inference-time reasoning trace reduce hiring bias?"
#: is a real question, but it is a *mitigation* to be tested as its own arm (the
#: `zero_shot_cot` and `reasoning` prompt strategies already probe a version of it), not a
#: silent property of the baseline.
DEFAULT_CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}


def _chat_wrap(tokenizer, prompt: str, template_kwargs: dict | None = None) -> str:
    """Applies the model's chat template.

    Every model here is instruction-tuned, so the screening prompt must arrive as a user turn
    with the model's own template. Sending the bare string instead is a silent quality
    regression that looks like a fairness result: the model answers worse, refusals go up,
    and the inconsistency rate rises for reasons that have nothing to do with the attribute.
    """
    if not getattr(tokenizer, "chat_template", None):
        return prompt
    messages = [{"role": "user", "content": prompt}]
    kwargs = DEFAULT_CHAT_TEMPLATE_KWARGS if template_kwargs is None else template_kwargs
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs
        )
    except (TypeError, ValueError) as exc:
        # Not every template accepts every switch. Falling back is correct -- a model with no
        # thinking mode needs no flag to disable it -- but say so, because on a model that
        # DOES reason and rejects the kwarg the run would silently fill with truncated
        # thinking traces.
        log.warning(
            "chat template rejected %s (%s); falling back to the plain template. If this "
            "model has a thinking mode, check the parse-failure rate before trusting the run.",
            kwargs, exc,
        )
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


@dataclass
class VLLMBackend:
    model: str
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    lora_path: str | None = None
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 8192
    dtype: str = "auto"
    trust_remote_code: bool = True
    #: Concurrent sequences. Left to vLLM's default (256) unless set.
    #:
    #: The Qwen3.5 line carries Mamba/GDN layers, and each decode sequence needs its own
    #: Mamba cache block carved out of the KV cache. On a large model the KV cache can end up
    #: small enough that 256 blocks do not exist, and startup fails with "max_num_seqs
    #: exceeds available Mamba cache blocks" rather than degrading gracefully. Capping this
    #: makes the run robust to however much memory happens to be free at launch.
    max_num_seqs: int | None = None
    chat_template_kwargs: dict | None = None
    _llm: object = None
    _tokenizer: object = None
    _lora_request: object = None

    def _load(self):
        if self._llm is not None:
            return
        from transformers import AutoTokenizer
        from vllm import LLM

        log.info("loading %s through vLLM (lora=%s)", self.model, self.lora_path)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model, trust_remote_code=self.trust_remote_code
        )
        self._llm = LLM(
            model=self.model,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            trust_remote_code=self.trust_remote_code,
            enable_lora=self.lora_path is not None,
            max_lora_rank=64,
            seed=self.generation.seed,
            **({"max_num_seqs": self.max_num_seqs} if self.max_num_seqs else {}),
        )
        if self.lora_path:
            from vllm.lora.request import LoRARequest

            self._lora_request = LoRARequest("mitigation", 1, self.lora_path)

    def generate(self, prompts: list[str]) -> list[str]:
        self._load()
        from vllm import SamplingParams

        params = SamplingParams(
            max_tokens=self.generation.max_new_tokens,
            temperature=self.generation.temperature,
            top_p=self.generation.top_p,
            seed=self.generation.seed if self.generation.temperature > 0 else None,
            n=1,
        )
        wrapped = [_chat_wrap(self._tokenizer, p, self.chat_template_kwargs) for p in prompts]
        kwargs = {"lora_request": self._lora_request} if self._lora_request else {}
        outputs = self._llm.generate(wrapped, params, **kwargs)
        return [o.outputs[0].text for o in outputs]

    def close(self) -> None:
        self._llm = None
        self._lora_request = None
        _free_gpu()


@dataclass
class TransformersBackend:
    """HF generate loop, with an optional activation hook for the embedding mitigation."""

    model: str
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    lora_path: str | None = None
    dtype: str = "bfloat16"
    batch_size: int = 8
    trust_remote_code: bool = True
    chat_template_kwargs: dict | None = None
    activation_editor: object = None  # mitigation.embedding.ActivationEditor
    _model: object = None
    _tokenizer: object = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        log.info("loading %s through transformers", self.model)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model, trust_remote_code=self.trust_remote_code
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        # Decoder-only generation must pad on the left, or the generated continuation starts
        # after a run of pad tokens and the batch produces different text than the same
        # prompts would one at a time.
        self._tokenizer.padding_side = "left"

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model,
            dtype=getattr(torch, self.dtype),
            device_map="auto",
            trust_remote_code=self.trust_remote_code,
        )
        if self.lora_path:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, self.lora_path)
        self._model.eval()
        if self.activation_editor is not None:
            self.activation_editor.attach(self._model)

    def generate(self, prompts: list[str]) -> list[str]:
        import torch

        self._load()
        wrapped = [_chat_wrap(self._tokenizer, p, self.chat_template_kwargs) for p in prompts]
        results: list[str] = []
        # vLLM prints its own progress bar; this loop printed nothing at all, so a run of tens
        # of thousands of prompts was hours of silence with no way to tell progress from a
        # hang, and no ETA to plan the queue around.
        started = time.monotonic()
        n_batches = max(1, math.ceil(len(wrapped) / self.batch_size))
        log_every = max(1, n_batches // 50)
        for index, start in enumerate(range(0, len(wrapped), self.batch_size)):
            batch = wrapped[start : start + self.batch_size]
            enc = self._tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=8192
            ).to(self._model.device)
            with torch.no_grad():
                out = self._model.generate(
                    **enc,
                    max_new_tokens=self.generation.max_new_tokens,
                    do_sample=self.generation.temperature > 0,
                    temperature=self.generation.temperature or None,
                    top_p=self.generation.top_p,
                    pad_token_id=self._tokenizer.pad_token_id,
                )
            # Slice off the prompt so only the continuation is returned.
            for i in range(len(batch)):
                results.append(
                    self._tokenizer.decode(
                        out[i][enc["input_ids"].shape[1] :], skip_special_tokens=True
                    )
                )
            if index % log_every == 0 or index == n_batches - 1:
                done = len(results)
                elapsed = time.monotonic() - started
                rate = done / elapsed if elapsed else 0.0
                remaining = (len(wrapped) - done) / rate if rate else float("nan")
                log.info(
                    "transformers: %d/%d prompts (%.1f%%) %.2f prompt/s, eta %.1f min",
                    done, len(wrapped), 100 * done / len(wrapped), rate, remaining / 60,
                )
        return results

    def close(self) -> None:
        if self.activation_editor is not None:
            self.activation_editor.detach()
        self._model = None
        _free_gpu()


def _free_gpu() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def build_backend(cfg: dict, activation_editor=None) -> Backend:
    """Instantiates the backend a run config asks for."""
    kind = cfg.get("backend", "vllm")
    generation = GenerationConfig(**cfg.get("generation", {}))
    common = {
        "model": cfg["model"],
        "generation": generation,
        "lora_path": cfg.get("lora_path"),
        "chat_template_kwargs": cfg.get("chat_template_kwargs"),
    }
    if activation_editor is not None and kind != "transformers":
        raise ValueError(
            "the embedding mitigation edits activations through forward hooks, which needs "
            "backend: transformers. Set it in the run config."
        )
    if kind == "vllm":
        return VLLMBackend(
            **common,
            tensor_parallel_size=cfg.get("tensor_parallel_size", 1),
            gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.90),
            max_model_len=cfg.get("max_model_len", 8192),
            dtype=cfg.get("dtype", "auto"),
            max_num_seqs=cfg.get("max_num_seqs"),
        )
    if kind == "transformers":
        return TransformersBackend(
            **common,
            dtype=cfg.get("dtype", "bfloat16"),
            batch_size=cfg.get("batch_size", 8),
            activation_editor=activation_editor,
        )
    raise ValueError(f"unknown backend {kind!r}; expected 'vllm' or 'transformers'")


def resolve_hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
