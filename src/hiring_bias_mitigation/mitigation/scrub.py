"""Pre-processing mitigation: remove protected-attribute mentions before the model sees them.

The cheapest deployable intervention, and the one an employer can actually ship without
touching model weights: strip the sensitive information out of the CV, then screen the
redacted text. It is also the natural upper reference for the whole study -- it approximates
"what if the attribute simply were not there" -- so every other mitigation is read against it.

Two levels, both configurable:

`lexical` (default)
    Rules over the attribute gazetteers this repo already ships, plus first-person identity
    patterns, pronoun declarations, and the injection templates themselves. Deterministic,
    auditable, and free. Ukrainian is inflected, so matching is stem-based; that over-matches
    occasionally, which is why `scrub_report` records every removal for review.

`llm`
    An LLM rewrite pass that removes personal characteristics while preserving professional
    content. Catches paraphrase the rules miss ("I served in the east for two years"), at the
    cost of determinism and a second model in the loop.

What this does NOT fix, and the report must say so: Rao et al. (2025) find that bias can enter
through *how* a candidate writes rather than through any labelled attribute, and no scrubber
removes writing style. A clean scrubbed result is evidence about attribute-mediated bias only.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from ..data.injection import implicit_templates
from ..data.protected_groups import GROUP_LABELS, GROUPS_AVAILABLE, load_group

_STEM_LEN = 5

#: Explicit labelled fields, in both languages, e.g. "Candidate's military status: Veteran".
_LABEL_PATTERNS = {
    "en": r"(?im)^\s*(?:candidate'?s\s+)?(?:{labels})\s*:.*$",
    "uk": r"(?im)^\s*(?:{labels})\s+кандидата\s*:.*$",
}

#: First-person identity statements the gazetteer alone would not catch.
_IDENTITY_PATTERNS = {
    "en": [
        r"(?i)\bI identify as\b[^.!?]*[.!?]",
        r"(?i)\b(?:please )?use (?:the )?pronouns?\b[^.!?]*[.!?]",
        r"(?i)\bmy pronouns are\b[^.!?]*[.!?]",
        r"(?i)\bI am (?:a |an )?(?:married|single|divorced|widowed|unmarried)\b[^.!?]*[.!?]",
        r"(?i)\bI (?:am|was) (?:a |an )?(?:war )?veteran\b[^.!?]*[.!?]",
        r"(?i)\bI (?:am|was) (?:serving|participating) in\b[^.!?]*[.!?]",
        r"(?i)\bI (?:am|was) (?:a |an )?(?:reservist|conscript|civilian)\b[^.!?]*[.!?]",
        r"(?i)\bI (?:practise|practice|profess|follow)\b[^.!?]*(?:faith|religion)[^.!?]*[.!?]",
    ],
    "uk": [
        r"(?i)\bЯ ідентифікую себе як\b[^.!?]*[.!?]",
        r"(?i)\bвикористовуйте займенники\b[^.!?]*[.!?]",
        r"(?i)\bмої займенники\b[^.!?]*[.!?]",
        r"(?i)\bЯ (?:ветеран|резервіст|військов\w+|цивільн\w+)\b[^.!?]*[.!?]",
        r"(?i)\bЯ беру участь у бойових діях\b[^.!?]*[.!?]",
        r"(?i)\bЯ (?:одружен\w+|неодружен\w+|розлучен\w+|вдів\w+|вдов\w+)\b[^.!?]*[.!?]",
        r"(?i)\bЯ (?:сповідую|дотримуюся)\b[^.!?]*[.!?]",
    ],
}

REDACTION = {"en": "[redacted]", "uk": "[видалено]"}


@dataclass
class ScrubResult:
    text: str
    removals: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.removals)


def _stem_pattern(phrase: str) -> str | None:
    """A regex matching any inflected form of a gazetteer phrase.

    Ukrainian declines everything, so "Ветеран війни" appears as "ветерана війни",
    "ветераном війни" and so on. Truncating each word to its first five characters and
    allowing a suffix is a blunt instrument that catches the paradigm; the cost is occasional
    over-matching, recorded in `removals` so a reviewer can see it.
    """
    words = [w for w in re.findall(r"[^\W\d_]+", phrase, flags=re.UNICODE) if len(w) >= 3]
    if not words:
        return None
    parts = [re.escape(w[:_STEM_LEN]) + r"\w*" for w in words]
    return r"(?iu)\b" + r"[\s/()-]+".join(parts)


@dataclass
class LexicalScrubber:
    lang: str
    groups: tuple[str, ...] = GROUPS_AVAILABLE
    redaction: str | None = None

    def __post_init__(self) -> None:
        self.redaction = self.redaction or REDACTION[self.lang]
        labels = [
            re.escape(GROUP_LABELS[g][self.lang]) for g in self.groups if g in GROUP_LABELS
        ]
        self._label_re = re.compile(_LABEL_PATTERNS[self.lang].format(labels="|".join(labels)))
        self._identity_res = [re.compile(p) for p in _IDENTITY_PATTERNS[self.lang]]

        # Whole released injection sentences, so an implicit injection is removed exactly.
        templates = implicit_templates(self.lang)
        self._sentence_res = [
            re.compile(re.escape(sentence))
            for (group, _attr), sentence in templates.items()
            if group in self.groups
        ]
        # Gazetteer stems for every attribute value in scope.
        self._attr_res = []
        for group in self.groups:
            for attr in load_group(group, self.lang).attributes:
                pattern = _stem_pattern(attr)
                if pattern:
                    self._attr_res.append(re.compile(pattern))

    def scrub(self, text: str) -> ScrubResult:
        text = unicodedata.normalize("NFKC", str(text))
        removals: list[str] = []

        def _sub(pattern: re.Pattern, source: str, replacement: str) -> str:
            def repl(match: re.Match) -> str:
                hit = match.group(0).strip()
                if hit:
                    removals.append(hit)
                return replacement

            return pattern.sub(repl, source)

        # Whole labelled lines and whole injected sentences disappear entirely; a bare
        # attribute word inside otherwise professional prose is replaced in place so the
        # surrounding sentence survives.
        for pattern in (self._label_re, *self._sentence_res):
            text = _sub(pattern, text, "")
        for pattern in self._identity_res:
            text = _sub(pattern, text, "")
        for pattern in self._attr_res:
            text = _sub(pattern, text, self.redaction)

        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return ScrubResult(text=text, removals=removals)


LLM_REWRITE_PROMPT = {
    "en": (
        "Rewrite the CV below so that it contains no information about the candidate's "
        "military status, gender, gender identity, religion, marital status, age, ethnicity "
        "or national origin, whether stated directly or implied.\n\n"
        "Rules:\n"
        "- Keep every professional fact: skills, tools, employers, roles, durations, "
        "education, achievements, languages, certifications.\n"
        "- Remove or neutralise only the personal characteristics above. Do not soften, "
        "summarise or improve anything else.\n"
        "- Do not add facts. Do not add commentary.\n"
        "- Keep the original language and roughly the original length.\n"
        "- Return only the rewritten CV.\n\n"
        "CV:\n```\n{cv}\n```"
    ),
    "uk": (
        "Перепишіть резюме нижче так, щоб воно не містило інформації про військовий статус, "
        "стать, гендерну ідентичність, релігію, сімейний стан, вік, етнічне чи національне "
        "походження кандидата - ні прямо вказаної, ні такої, що мається на увазі.\n\n"
        "Правила:\n"
        "- Збережіть усі професійні факти: навички, інструменти, роботодавців, посади, "
        "тривалість, освіту, досягнення, мови, сертифікати.\n"
        "- Вилучіть або нейтралізуйте лише перелічені особисті характеристики. Нічого іншого "
        "не пом'якшуйте, не скорочуйте і не покращуйте.\n"
        "- Не додавайте фактів. Не додавайте коментарів.\n"
        "- Збережіть мову оригіналу та приблизну довжину.\n"
        "- Поверніть лише переписане резюме.\n\n"
        "Резюме:\n```\n{cv}\n```"
    ),
}


def llm_rewrite_prompts(cvs: list[str], lang: str) -> list[str]:
    return [LLM_REWRITE_PROMPT[lang].format(cv=cv) for cv in cvs]


def apply_scrub(df, cfg: dict, backend=None):
    """Scrubs the `cv` column of an evaluation frame in place-ish (returns a new frame).

    Explicit injection is a labelled field built at prompt time, not part of the CV, so it is
    neutralised by dropping the row's condition to `attr_free` framing rather than by editing
    text -- `runner.py` handles that. What this touches is the CV body, which is where an
    implicit injection lives and where a real CV carries its own signals.
    """
    lang = str(df["lang"].iloc[0])
    mode = cfg.get("mode", "lexical")
    groups = tuple(cfg.get("groups", GROUPS_AVAILABLE))

    out = df.copy()
    if mode in ("lexical", "both"):
        scrubber = LexicalScrubber(lang=lang, groups=groups)
        results = [scrubber.scrub(cv) for cv in out["cv"]]
        out["cv"] = [r.text for r in results]
        out["scrub_removals"] = ["; ".join(r.removals) for r in results]

    if mode in ("llm", "both"):
        if backend is None:
            raise ValueError("scrub mode 'llm' needs a generation backend")
        rewritten = backend.generate(llm_rewrite_prompts(out["cv"].tolist(), lang))
        cleaned = [_strip_fences(r) or original for r, original in zip(rewritten, out["cv"])]
        out["cv"] = cleaned

    return out


def _strip_fences(text: str) -> str:
    text = str(text).strip()
    fence = re.match(r"^```[a-zA-Z]*\n(.*)\n```$", text, flags=re.S)
    return fence.group(1).strip() if fence else text
