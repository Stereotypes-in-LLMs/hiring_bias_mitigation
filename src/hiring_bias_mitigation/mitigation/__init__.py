"""Mitigation families, and where in the pipeline each one intervenes.

    family      intervenes at            costs                       artefact
    ----------  -----------------------  --------------------------  --------------------
    none        -                        -                           (baseline)
    prompt      prompt construction      0 (1 extra call for the     none
                                         two-pass verifier)
    scrub       the CV text, pre-prompt  0, or 1 rewrite call        none
    embedding   the residual stream      one fitting pass            eraser .npz
    sft         model weights            a training run              LoRA adapter
    dpo         model weights            SFT + a preference run      LoRA adapter

The families are deliberately comparable: every one is evaluated by the same
`eval/runner.py` over the same benchmark, and the only thing that changes between a baseline
run and a mitigated run is the block named here. That is what lets the report put a
zero-cost prompt edit and a two-stage training pipeline in the same table.
"""

FAMILIES = ("none", "prompt", "scrub", "embedding", "sft", "dpo")
