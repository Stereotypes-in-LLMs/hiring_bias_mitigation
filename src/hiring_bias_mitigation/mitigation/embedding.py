"""Representation-level mitigation: erase the protected attribute from the residual stream.

The intuition the audit paper's discussion points at (section 6.2): if the model's decision
moves when only the attribute changes, then somewhere in the forward pass the attribute is
represented, linearly enough for downstream layers to read. Erase that direction and the
decision has nothing attribute-shaped left to condition on. This is an inference-time
intervention -- no weights change, nothing is retrained -- so it is the cheapest thing that
can be deployed on top of a model an organisation does not control.

Three fitting methods, in increasing strength of guarantee:

`mean_diff`
    Direction = mean activation under attribute A minus mean under attribute B, over matched
    counterfactual pairs. One direction, ablated by projection. Crude but interpretable, and
    the differences are genuinely paired here, which removes most of the nuisance variance.

`inlp`
    Iterative nullspace projection: fit a linear probe for the attribute, project its
    direction out, refit on the projected representations, repeat. Removes a subspace rather
    than a line. Standard, but it can over-project -- each round strips capacity that may be
    carrying task-relevant information, which is why the utility column matters here.

`leace`
    Least-squares concept erasure (Belrose et al. 2023), closed form. Guarantees the concept
    is not linearly recoverable while making the least-squares-smallest change to the
    representation. The right default: it is the only one of the three with a guarantee on
    both sides, erasure and damage.

Caveats the report must carry. Erasure is *linear*: a concept encoded non-linearly survives
it. Fitting is done on the training pool, never on the benchmark, or the numbers are fitted
to their own test set. And a projection applied at every token position changes generation
quality, so the utility column is not optional here -- it is how you find out whether the
model still works.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..utils.logging import get_logger

log = get_logger(__name__)

METHODS = ("mean_diff", "inlp", "leace")


# ---------------------------------------------------------------------------------------
# Fitting the erasure operator
# ---------------------------------------------------------------------------------------

@dataclass
class Eraser:
    """An affine erasure operator: x -> mean + P @ (x - mean)."""

    projection: np.ndarray  # (d, d)
    mean: np.ndarray  # (d,)
    method: str
    layers: tuple[int, ...]
    n_fitted: int
    rank_removed: int

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            projection=self.projection,
            mean=self.mean,
            method=self.method,
            layers=np.array(self.layers),
            n_fitted=self.n_fitted,
            rank_removed=self.rank_removed,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> Eraser:
        data = np.load(path, allow_pickle=False)
        return cls(
            projection=data["projection"],
            mean=data["mean"],
            method=str(data["method"]),
            layers=tuple(int(x) for x in data["layers"]),
            n_fitted=int(data["n_fitted"]),
            rank_removed=int(data["rank_removed"]),
        )


def _orthogonal_projection_onto(matrix: np.ndarray) -> np.ndarray:
    """P = M (M^T M)^+ M^T -- orthogonal projection onto the column space of M."""
    gram = matrix.T @ matrix
    return matrix @ np.linalg.pinv(gram) @ matrix.T


def fit_mean_diff(activations: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, int]:
    """Ablate the single direction separating the two label groups."""
    classes = np.unique(labels)
    if classes.size < 2:
        raise ValueError("mean_diff needs at least two attribute classes")
    # With >2 attributes, every class is contrasted against the pooled remainder and the
    # resulting directions are stacked; ablating their span generalises the binary case.
    directions = []
    for cls in classes:
        mask = labels == cls
        direction = activations[mask].mean(axis=0) - activations[~mask].mean(axis=0)
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            directions.append(direction / norm)
    basis = np.stack(directions, axis=1)  # (d, k)
    projection = np.eye(activations.shape[1]) - _orthogonal_projection_onto(basis)
    return projection, int(np.linalg.matrix_rank(basis))


def fit_inlp(
    activations: np.ndarray, labels: np.ndarray, n_iterations: int = 8, seed: int = 42
) -> tuple[np.ndarray, int]:
    """Iterative nullspace projection with linear probes."""
    from sklearn.linear_model import SGDClassifier

    d = activations.shape[1]
    projection = np.eye(d)
    current = activations.copy()
    removed = 0

    for it in range(n_iterations):
        probe = SGDClassifier(
            loss="log_loss", max_iter=2000, tol=1e-4, random_state=seed + it, alpha=1e-4
        )
        probe.fit(current, labels)
        accuracy = probe.score(current, labels)
        majority = float(np.bincount(_as_int(labels)).max()) / len(labels)
        # Stop once the probe is no better than always predicting the majority class -- the
        # concept is gone and further rounds would only destroy unrelated capacity.
        if accuracy <= majority + 0.01:
            log.info("INLP converged after %d rounds (probe acc %.3f)", it, accuracy)
            break
        weights = probe.coef_.T  # (d, n_classes) or (d, 1)
        step = np.eye(d) - _orthogonal_projection_onto(weights)
        projection = step @ projection
        current = current @ step.T
        removed += int(np.linalg.matrix_rank(weights))

    return projection, removed


def fit_leace(activations: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, int]:
    """Closed-form LEACE: the least-squares-optimal linear concept eraser.

    Concept labels are one-hot encoded and centred. Whitening before projecting is what makes
    the erasure optimal rather than merely sufficient -- projecting in the raw space removes
    more of the representation than the concept actually occupies.
    """
    x = activations - activations.mean(axis=0, keepdims=True)
    z = _one_hot(labels)
    z = z - z.mean(axis=0, keepdims=True)

    n = x.shape[0]
    cov_xx = (x.T @ x) / max(n - 1, 1)
    cov_xz = (x.T @ z) / max(n - 1, 1)

    whiten, unwhiten = _inv_sqrt_pair(cov_xx)
    projected = _orthogonal_projection_onto(whiten @ cov_xz)
    projection = np.eye(x.shape[1]) - unwhiten @ projected @ whiten
    return projection, int(np.linalg.matrix_rank(cov_xz))


def _inv_sqrt_pair(cov: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Returns (Sigma^-1/2, Sigma^+1/2) via eigendecomposition of a symmetric matrix."""
    cov = (cov + cov.T) / 2
    values, vectors = np.linalg.eigh(cov)
    values = np.clip(values, eps, None)
    inv_sqrt = vectors @ np.diag(values ** -0.5) @ vectors.T
    sqrt = vectors @ np.diag(values ** 0.5) @ vectors.T
    return inv_sqrt, sqrt


def _as_int(labels: np.ndarray) -> np.ndarray:
    _, inverse = np.unique(labels, return_inverse=True)
    return inverse


def _one_hot(labels: np.ndarray) -> np.ndarray:
    codes = _as_int(labels)
    out = np.zeros((codes.size, codes.max() + 1), dtype=float)
    out[np.arange(codes.size), codes] = 1.0
    return out


def fit_eraser(
    activations: np.ndarray,
    labels: np.ndarray,
    method: str,
    layers: tuple[int, ...],
    inlp_iterations: int = 8,
    seed: int = 42,
) -> Eraser:
    activations = np.asarray(activations, dtype=np.float64)
    labels = np.asarray(labels)
    if activations.ndim != 2:
        raise ValueError(f"expected (n, d) activations, got shape {activations.shape}")
    if method == "mean_diff":
        projection, rank = fit_mean_diff(activations, labels)
    elif method == "inlp":
        projection, rank = fit_inlp(activations, labels, inlp_iterations, seed)
    elif method == "leace":
        projection, rank = fit_leace(activations, labels)
    else:
        raise ValueError(f"unknown erasure method {method!r}; expected one of {METHODS}")

    log.info(
        "fitted %s eraser on %d activations (d=%d), removed rank %d",
        method, activations.shape[0], activations.shape[1], rank,
    )
    return Eraser(
        projection=projection.astype(np.float32),
        mean=activations.mean(axis=0).astype(np.float32),
        method=method,
        layers=tuple(layers),
        n_fitted=int(activations.shape[0]),
        rank_removed=rank,
    )


def linear_probe_accuracy(activations: np.ndarray, labels: np.ndarray, seed: int = 42) -> float:
    """How recoverable the attribute is from these representations.

    Reported before and after erasure. Before-accuracy near chance means there was no linear
    attribute signal to remove at this layer, and any fairness change from the intervention
    needs a different explanation; after-accuracy near chance is the erasure working.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    probe = LogisticRegression(max_iter=2000, random_state=seed)
    scores = cross_val_score(probe, activations, labels, cv=min(5, len(np.unique(labels)) + 2))
    return float(scores.mean())


# ---------------------------------------------------------------------------------------
# Applying it during generation
# ---------------------------------------------------------------------------------------

@dataclass
class ActivationEditor:
    """Applies an eraser to selected decoder layers through forward hooks."""

    eraser: Eraser
    strength: float = 1.0
    _handles: list = field(default_factory=list)
    _torch_projection: object = None
    _torch_mean: object = None

    def attach(self, model) -> None:
        import torch

        layers = _decoder_layers(model)
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        self._torch_projection = torch.tensor(
            self.eraser.projection, device=device, dtype=torch.float32
        )
        self._torch_mean = torch.tensor(self.eraser.mean, device=device, dtype=torch.float32)

        for index in self.eraser.layers:
            if index >= len(layers):
                raise IndexError(
                    f"eraser was fitted for layer {index} but the model has {len(layers)}"
                )
            self._handles.append(
                layers[index].register_forward_hook(self._make_hook(dtype))
            )
        log.info(
            "attached %s eraser (strength %.2f) to layers %s",
            self.eraser.method, self.strength, list(self.eraser.layers),
        )

    def _make_hook(self, dtype):
        import torch

        def hook(_module, _inputs, output):
            # A decoder layer returns either a tensor or a tuple whose first element is the
            # hidden state; editing the wrong element silently does nothing.
            hidden = output[0] if isinstance(output, tuple) else output
            with torch.no_grad():
                original = hidden.to(torch.float32)
                centred = original - self._torch_mean
                erased = self._torch_mean + centred @ self._torch_projection.T
                blended = original + self.strength * (erased - original)
                edited = blended.to(dtype)
            if isinstance(output, tuple):
                return (edited, *tuple(output[1:]))
            return edited

        return hook

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _decoder_layers(model):
    """Finds the decoder-layer list across the naming conventions in use.

    Llama/Qwen/Mistral expose `model.model.layers`; a PEFT-wrapped model adds another level;
    Gemma's multimodal checkpoints put the text stack under `model.language_model`.
    """
    candidates = (
        "model.layers",
        "model.model.layers",
        "base_model.model.model.layers",
        "model.language_model.layers",
        "model.model.language_model.layers",
        "transformer.h",
    )
    for path in candidates:
        node = model
        for part in path.split("."):
            node = getattr(node, part, None)
            if node is None:
                break
        if node is not None and hasattr(node, "__len__") and len(node):
            return node
    raise AttributeError(
        f"could not locate the decoder layer list on {type(model).__name__}. "
        f"Tried: {candidates}"
    )


def collect_activations(
    model, tokenizer, prompts: list[str], layers: tuple[int, ...], batch_size: int = 8
) -> np.ndarray:
    """Mean-pooled hidden states at the requested layers, one vector per prompt.

    Pooling is over non-padding positions of the prompt: the decision has not been generated
    yet, so the attribute's representation is whatever the prompt itself induced. When several
    layers are requested their pooled states are averaged, which keeps one operator per run
    instead of one per layer.
    """
    import torch

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    collected: list[np.ndarray] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=8192
        ).to(model.device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        mask = enc["attention_mask"].unsqueeze(-1).to(torch.float32)
        # hidden_states[0] is the embedding output, so layer i sits at index i + 1.
        stacked = torch.stack([out.hidden_states[i + 1].to(torch.float32) for i in layers])
        pooled = (stacked * mask).sum(dim=2) / mask.sum(dim=1).clamp(min=1)
        collected.append(pooled.mean(dim=0).cpu().numpy())
    return np.concatenate(collected, axis=0)
