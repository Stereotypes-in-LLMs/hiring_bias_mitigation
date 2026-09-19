"""SFT on counterfactually-augmented data.

The `sft` family on its own is one of the study's arms -- the user-facing question is whether
supervised fine-tuning alone closes the disparity, or whether the preference stage is doing
the work. So this is a complete mitigation, not a warm-up for DPO, and it is evaluated as
one. The `dpo` configs then point their `base_model` at an SFT run's output directory, which
makes SFT-only and SFT-then-DPO a clean ablation over the identical data.

What the objective actually installs: the training targets are attribute-invariant by
construction (same verdict across every attribute variant of a profile, rationale never
naming the attribute), so cross-entropy on them is a direct push toward counterfactual
consistency -- the thing the inconsistency rate measures. What it cannot install is a
*preference* between an invariant and a biased response, because it never sees a biased one.
That is exactly the gap the DPO arm tests.

Run:
    python -m hiring_bias_mitigation.mitigation.sft --config configs/mitigation/sft/<x>.yaml
"""

from __future__ import annotations

import argparse

from ..eval.backends import DEFAULT_CHAT_TEMPLATE_KWARGS
from ..utils.config import load_config
from ..utils.logging import get_logger
from ..utils.seed import set_seed
from . import training_common as C

log = get_logger(__name__)


def render_chat(dataset, tokenizer, chat_template_kwargs: dict | None = None):
    """Applies the chat template, splitting the result into a prompt/completion text pair.

    Three constraints have to hold at once, and this is the only shape that satisfies all of
    them.

    *The template must be applied.* The audit prompts every model through its chat template;
    training on bare strings and evaluating through the template is a mismatch that surfaces
    later as the mitigation "not working".

    *The loss must cover the completion only.* The prompt is a job ad and a CV the model is
    shown, not text it should learn to produce. TRL 1.x refuses a `formatting_func` alongside
    `completion_only_loss` -- a formatter flattens the pair into one language-modelling string
    and the mask can no longer tell the halves apart.

    *The split must be exact at the token level.* The completion is taken as the **string
    remainder** of the full render after the prompt render, so concatenating the two
    reproduces the template's own output byte for byte. Handing TRL the message lists instead
    makes it tokenise the prompt separately, and Qwen's template emits `<think>\n` when asked
    for a generation prompt against `<think>\n\n</think>\n\n` in the full render -- the same
    string, different tokens, which TRL reports as "Mismatch between tokenized prompt and the
    start of tokenized prompt+completion" and then masks on a guess.

    `chat_template_kwargs` defaults to the audit's, because a difference here is a difference
    between what the model is trained to emit and what it is later asked for.
    """
    kwargs = dict(chat_template_kwargs or DEFAULT_CHAT_TEMPLATE_KWARGS)

    def render(row: dict) -> dict:
        user = [{"role": "user", "content": row["prompt"]}]
        assistant = [{"role": "assistant", "content": row["completion"]}]
        prompt = tokenizer.apply_chat_template(
            user, tokenize=False, add_generation_prompt=True, **kwargs
        )
        full = tokenizer.apply_chat_template(user + assistant, tokenize=False, **kwargs)
        if not full.startswith(prompt):
            raise ValueError(
                "the chat template does not render the prompt as a prefix of "
                "prompt+completion, so completion-only loss cannot be masked correctly. "
                f"Prompt ends {prompt[-60:]!r}; full has {full[len(prompt) - 60:len(prompt)]!r}"
            )
        return {"prompt": prompt, "completion": full[len(prompt):]}

    return dataset.map(
        render,
        remove_columns=[c for c in dataset.column_names if c not in ("prompt", "completion")],
    )


def decision_weighted_loss(weight: float, n_decision_tokens: int):
    """Cross-entropy that counts the decision tokens more than the rationale.

    Measured on this data: a completion is ~38 tokens, of which the `decision` field is 6.
    Plain token-level cross-entropy therefore puts **16% of the gradient on the thing the
    audit measures and 71% on the prose around it** — and the first SFT run showed exactly
    that split in its results, halving the rate at which the model names a protected attribute
    while leaving every acceptance-rate disparity untouched. The model learned to write like
    the teacher without learning to decide like it.

    The decision is always the first field of the JSON object, so the first
    `n_decision_tokens` unmasked positions of each row carry it. Weighting by position rather
    than by searching for a field name keeps this independent of tokeniser quirks and of the
    language the rationale is written in.
    """
    import torch
    import torch.nn.functional as F

    def loss_fn(outputs, labels, num_items_in_batch=None, **kwargs):
        logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
        vocab = logits.shape[-1]

        # Shift the *labels*, never the logits. The logits are [batch, seq, ~250k] and the
        # obvious `logits[..., :-1, :].contiguous()` plus a transpose for cross_entropy makes
        # two further copies of that tensor per step, at a different size every batch under
        # dynamic padding. That fed allocator fragmentation which grew for eight hours until
        # the whole machine ran out of memory. Padding the (tiny) label tensor by one and
        # viewing the logits flat gives the same next-token alignment with no copy at all.
        shifted = F.pad(labels, (0, 1), value=-100)[..., 1:]
        per_token = F.cross_entropy(
            logits.view(-1, vocab), shifted.reshape(-1), reduction="none", ignore_index=-100
        ).view_as(shifted)
        supervised = shifted != -100

        # Rank each supervised position within its row; the first n are the decision field.
        order = torch.cumsum(supervised.to(torch.int32), dim=-1)
        is_decision = supervised & (order <= n_decision_tokens)

        weights = torch.where(
            is_decision, torch.full_like(per_token, weight), torch.ones_like(per_token)
        ) * supervised
        total = (per_token * weights).sum()
        denominator = weights.sum().clamp(min=1.0)
        return total / denominator

    return loss_fn


def decision_token_count(tokenizer, sample: str = '{"decision": "reject"') -> int:
    """How many tokens the decision field occupies, measured rather than assumed."""
    return len(tokenizer(sample, add_special_tokens=False)["input_ids"])


def train(cfg: dict) -> C.TrainArtifacts:
    from trl import SFTConfig, SFTTrainer

    set_seed(cfg.get("seed", 42))
    C.wandb_init(cfg)
    output_dir = C.resolve_run_dir(cfg)

    # `data_kind` lets a variant arm read a different split without a new loader.
    splits = C.load_generated_dataset(cfg, cfg.get("data_kind", "sft"))
    train_ds = splits["train"]
    eval_ds = splits.get("validation")

    model, tokenizer = C.load_model_and_tokenizer(cfg)
    peft = C.peft_config(cfg)

    base_args = C.training_arguments(
        cfg, output_dir, has_eval=eval_ds is not None, n_train=len(train_ds)
    )
    weight = float(cfg.get("decision_loss_weight", 1.0))

    sft_args = SFTConfig(
        **base_args.to_dict(),
        **C.supported_kwargs(
            SFTConfig,
            {
                "max_length": cfg.get("max_seq_len", 2048),
                "packing": False,
                # Train on the response only. The prompt is a job ad and a CV the model is
                # shown, not text it should learn to produce.
                "completion_only_loss": True,
                # TRL's default `chunked_nll` computes entropy from fields its own patched
                # forward attaches (`outputs.num_valid_tokens`). A custom loss bypasses that
                # forward, so the entropy branch then reads an attribute that is not there and
                # the run dies on the first training step. `nll` is the same maths without the
                # chunked lm_head projection, and it does not touch those fields.
                "loss_type": "nll" if weight != 1.0 else cfg.get("loss_type"),
            },
            "SFTConfig",
        ),
    )

    chat_kwargs = cfg.get("chat_template_kwargs")
    loss_fn = None
    if weight != 1.0:
        n_tokens = decision_token_count(tokenizer)
        log.info("weighting the first %d completion tokens (the decision) x%.1f",
                 n_tokens, weight)
        loss_fn = decision_weighted_loss(weight, n_tokens)

    train_ds = render_chat(train_ds, tokenizer, chat_kwargs)
    if eval_ds is not None:
        eval_ds = render_chat(eval_ds, tokenizer, chat_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft,
        callbacks=C.early_stopping_callbacks(cfg, eval_ds is not None)
            + C.memory_guard_callbacks(cfg),
        compute_loss_func=loss_fn,
    )

    log.info(
        "SFT %s on %d rows (%s)",
        cfg["base_model"], len(train_ds), "LoRA" if peft else "full fine-tune",
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    C.write_run_manifest(
        cfg, output_dir,
        {"mitigation_family": "sft", "n_train": len(train_ds),
         "n_eval": len(eval_ds) if eval_ds else 0},
    )
    return C.TrainArtifacts(output_dir, len(train_ds), len(eval_ds) if eval_ds else 0)


def main() -> None:
    C.configure_cuda_allocator()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg.setdefault("stage", "sft")
    artifacts = train(cfg)
    log.info("SFT complete -> %s", artifacts.output_dir)


if __name__ == "__main__":
    main()
