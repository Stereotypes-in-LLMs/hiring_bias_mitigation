"""Sets which configs are enabled in one runner script, by exact path.

`generate_experiment_configs.py --enable <preset>` applies a preset across every runner. This
does the narrower thing the mitigation plan needs: enable exactly these configs in exactly
this runner, leaving everything else commented out.

    python scripts/select_configs.py --runner scripts/run_all_prompt.sh \\
        configs/mitigation/prompt/qwen3.5-4b_uk_fairness_constitution.yaml \\
        configs/mitigation/prompt/qwen3.5-4b_uk_structured_rubric.yaml

    python scripts/select_configs.py --runner scripts/run_all_sft.sh --list
    python scripts/select_configs.py --runner scripts/run_all_sft.sh --none

Refuses to touch a runner whose sweep is currently running: bash reads a script incrementally
as it executes it, so rewriting one mid-run can corrupt the rest of the queue.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiring_bias_mitigation.utils.config import REPO_ROOT  # noqa: E402
from hiring_bias_mitigation.utils.logging import get_logger  # noqa: E402

log = get_logger("select_configs")

CONFIGS_BLOCK = re.compile(r"(CONFIGS=\()(.*?)(\n\))", re.S)


#: Interpreters that would be argv[0] when a runner is genuinely executing.
_SHELLS = {"bash", "sh", "dash", "zsh"}


def _is_running(runner: Path) -> bool:
    """Whether a shell is actually *executing* this runner.

    Deliberately not `pgrep -f <name>`. A substring match over full command lines also hits
    any process that merely *mentions* the script -- including the shell that invoked this
    very check, whose command line contains the name as an argument. That false positive
    makes the guard refuse every legitimate edit, which is worse than not having it.

    Instead: walk /proc, and treat a process as running the runner only when its argv[0] is a
    shell and the script path appears as one of its arguments. Our own process tree is
    excluded so the check cannot trip over itself.
    """
    resolved = str(runner.resolve())
    ours = {os.getpid(), os.getppid()}
    proc = Path("/proc")
    if not proc.exists():  # non-Linux: fall back to allowing the edit
        return False

    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) in ours:
            continue
        try:
            argv = (entry / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except (OSError, PermissionError):
            continue
        argv = [a for a in argv if a]
        if len(argv) < 2 or Path(argv[0]).name not in _SHELLS:
            continue
        for arg in argv[1:]:
            candidate = Path(arg)
            if arg == resolved or candidate.name == runner.name:
                try:
                    if str(candidate.resolve()) == resolved:
                        return True
                except OSError:
                    continue
    return False


def _entries(block: str) -> list[str]:
    """Every config path in the array, enabled or not, in order."""
    out = []
    for line in block.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped.startswith("configs/") and stripped.endswith(".yaml"):
            out.append(stripped.split()[0])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runner", required=True)
    parser.add_argument("configs", nargs="*")
    parser.add_argument("--list", action="store_true", help="show the array and exit")
    parser.add_argument("--none", action="store_true", help="comment everything out")
    parser.add_argument("--force", action="store_true",
                        help="edit even if this runner appears to be running (unsafe)")
    args = parser.parse_args()

    runner = Path(args.runner)
    if not runner.is_absolute():
        runner = REPO_ROOT / runner
    if not runner.exists():
        raise SystemExit(f"{runner} does not exist")

    text = runner.read_text(encoding="utf-8")
    match = CONFIGS_BLOCK.search(text)
    if not match:
        raise SystemExit(f"{runner}: no CONFIGS=( ... ) block found")
    available = _entries(match.group(2))

    if args.list:
        enabled = {
            line.strip().split()[0]
            for line in match.group(2).splitlines()
            if line.strip().startswith("configs/")
        }
        for config in available:
            print(("  [x] " if config in enabled else "  [ ] ") + config)
        print(f"\n{len(enabled)}/{len(available)} enabled in {runner.name}")
        return

    if not args.none and not args.configs:
        raise SystemExit("pass config paths, or --none, or --list")

    if _is_running(runner) and not args.force:
        raise SystemExit(
            f"{runner.name} appears to be running. Bash reads a script incrementally as it "
            f"executes, so rewriting it now can corrupt the rest of the queue.\n"
            f"Wait for it to finish, use scripts/queue_after.sh to append work, or pass "
            f"--force if you are certain."
        )

    wanted = set() if args.none else {c.lstrip("./") for c in args.configs}
    unknown = wanted - set(available)
    if unknown:
        raise SystemExit(
            f"{runner.name} has no entry for:\n"
            + "\n".join(f"  {c}" for c in sorted(unknown))
            + "\nRun `python scripts/generate_experiment_configs.py --no-runners` first."
        )

    body = "\n" + "\n".join(
        f"  {config}" if config in wanted else f"#   {config}" for config in available
    )
    text = text[: match.start(2)] + body + text[match.end(2) :]
    text = re.sub(
        r"# Currently enabled: \d+/\d+",
        f"# Currently enabled: {len(wanted)}/{len(available)}",
        text,
    )
    runner.write_text(text, encoding="utf-8")
    log.info("%s: %d/%d config(s) enabled", runner.name, len(wanted), len(available))
    for config in sorted(wanted):
        print(f"  [x] {config}")


if __name__ == "__main__":
    main()
