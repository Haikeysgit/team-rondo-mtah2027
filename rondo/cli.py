"""Command line: aggregates only, never rows."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

import pandas as pd

from rondo import kit
from rondo.io.load import check_bundle, load_match
from rondo.pipeline import run as run_pipeline
from rondo.post import summarize, validate_official, write_submission

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.toml"
RUN_MODULE = "rondo.cli"  # verify-repro runs ``python -m RUN_MODULE run ...`` in fresh processes


def _data_dir(value: str | None) -> Path | None:
    return Path(value) if value else None


def _check(args: argparse.Namespace) -> int:
    report = check_bundle(_data_dir(args.data_dir), expected_records=args.expected_records)
    for key, value in report.items():
        print(f"{key}: {value}")
    return 0


def _run(args: argparse.Namespace) -> int:
    from rondo import config as runtime_config

    data_dir = _data_dir(args.data_dir) or runtime_config.data_dir()
    with open(args.config, "rb") as handle:
        config = tomllib.load(handle)
    match = load_match(
        data_dir,
        cache=not args.no_cache,
        expected_frames=args.expected_frames,
        refresh=args.refresh_cache,
    )
    labels = run_pipeline(match, config)
    out = Path(args.out) if args.out else data_dir / "outputs" / "team-rondo.csv"
    digest = write_submission(labels, out)
    rows = args.validate_rows or match.meta.n_frames
    verdict = validate_official(out, rows=rows)

    summary = summarize(labels, match.meta)
    print(f"rows: {len(labels)}")
    for label in kit.pm.LABELS:
        count = summary["counts"][label]
        print(f"label {label}: {count} ({100.0 * count / len(labels):.2f}%)")
    print(f"possessions: {summary['possessions']}")
    print(f"mean possession length (s): {summary['mean_possession_s']:.2f}")
    print(f"sha256: {digest}")
    print("validator: ok" if verdict is None else f"validator: FAILED ({verdict})")
    return 0 if verdict is None else 1


def _verify_repro(args: argparse.Namespace) -> int:
    """``rondo run`` twice, each in a fresh process, into two files; compare their SHA-256.

    Run 1 is cold: it builds the match from the source files, as a judge's first run does, and
    rewrites the cache. Run 2 is warm: it reads that cache. Identical files therefore also show
    that the cache does not change the output.
    """
    base = [sys.executable, "-m", RUN_MODULE, "run", "--config", str(Path(args.config).resolve())]
    if args.data_dir:
        base += ["--data-dir", str(Path(args.data_dir).resolve())]
    for flag, value in (
        ("--expected-frames", args.expected_frames),
        ("--validate-rows", args.validate_rows),
    ):
        if value:
            base += [flag, str(value)]
    digests = []
    with tempfile.TemporaryDirectory(prefix="rondo-repro-", ignore_cleanup_errors=True) as tmp:
        for run, flags in ((1, ["--refresh-cache"]), (2, [])):
            out = Path(tmp) / f"run{run}" / "team-rondo.csv"
            done = subprocess.run(
                [*base, *flags, "--out", str(out)],
                check=False,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            if done.returncode != 0 or not out.is_file():
                reason = (done.stderr.strip().splitlines() or ["no output"])[-1]
                print(f"run {run}: failed (exit {done.returncode}): {reason}")
                return 1
            digests.append(hashlib.sha256(out.read_bytes()).hexdigest())
    print(f"run 1 sha256: {digests[0]}")
    print(f"run 2 sha256: {digests[1]}")
    same = digests[0] == digests[1]
    print("IDENTICAL" if same else "DIFFERENT")
    return 0 if same else 1


def _score(args: argparse.Namespace) -> int:
    from rondo import config as runtime_config
    from rondo.eval import load_windows, read_labels, score_all, write_result

    if args.result and not args.owner:
        _parser().error("--result needs --owner")
    data_dir = _data_dir(args.data_dir) or runtime_config.data_dir()
    try:
        windows = load_windows(data_dir)
    except ValueError as error:
        print(f"invalid window: {error}")
        return 1
    n_frames = len(windows["public"])
    if args.windows:
        unknown = sorted(set(args.windows) - set(windows))
        if unknown:
            print(f"unknown window(s): {', '.join(unknown)}; available: {', '.join(windows)}")
            return 2
        windows = {name: windows[name] for name in windows if name in args.windows}
    try:
        pred = read_labels(Path(args.csv), n_frames=n_frames)
    except ValueError as error:
        print(f"invalid submission: {error}")
        return 1

    try:
        table = score_all(pred, windows, bootstrap=args.bootstrap)
    except ValueError as error:  # two windows label the same frame differently
        print(f"invalid windows: {error}")
        return 1
    print(f"{'window':<10} {'graded':>7} {'spellF1':>8} {'90% interval':>15} {'macroF1':>8}  poss")
    for row in table.itertuples(index=False):
        interval = (
            f"{row.spell_bootstrap_low:.4f}..{row.spell_bootstrap_high:.4f}"
            if pd.notna(row.spell_bootstrap_low)
            else "-"
        )
        print(
            f"{row.window:<10} {row.n_graded:>7} {row.spell_f1_mean:>8.4f} {interval:>15} "
            f"{row.macro_f1:>8.4f}  {row.n_pred_spells} / {row.n_gt_spells}"
        )
    if args.result:
        command = f"rondo score {args.csv} --bootstrap {args.bootstrap}"
        if args.windows:
            command += f" --windows {' '.join(args.windows)}"
        write_result(Path(args.result), args.owner, command, table, args.notes)
        print(f"wrote {args.result}")
    return 0


def _label(args: argparse.Namespace) -> int:
    from rondo import config as runtime_config
    from rondo.labeltool.app import run as run_labeltool

    data_dir = _data_dir(args.data_dir) or runtime_config.data_dir()
    run_labeltool(data_dir, args.window, args.start, args.end)
    return 0


PLAY_MARGIN_FRAMES = 75  # ~3s either side of an error frame at 25fps


def _errors(args: argparse.Namespace) -> int:
    from rondo import config as runtime_config
    from rondo.eval import load_windows, read_labels
    from rondo.viz.errors import error_moments, format_timestamp

    data_dir = _data_dir(args.data_dir) or runtime_config.data_dir()
    windows = load_windows(data_dir)
    if args.window not in windows:
        print(f"unknown window {args.window!r}; available: {', '.join(windows)}")
        return 2
    truth = windows[args.window]
    try:
        pred = read_labels(Path(args.csv), n_frames=len(truth))
    except ValueError as error:
        print(f"invalid submission: {error}")
        return 1

    moments = error_moments(pred, truth)
    for status in ("missed", "spurious", "late"):
        print(f"{status}: {int((moments['status'] == status).sum())}")

    top = moments.head(20)
    for row in top.itertuples(index=False):
        change = f"{row.gt_from or row.pred_from} -> {row.gt_to or row.pred_to}"
        print(f"{format_timestamp(row.frame)}  frame {row.frame:>7}  {row.status:<8} {change}")

    if args.play:
        from rondo.labeltool.app import run as run_labeltool

        n_frames = len(truth)
        for row in top.itertuples(index=False):
            start = max(0, row.frame - PLAY_MARGIN_FRAMES)
            end = min(n_frames - 1, row.frame + PLAY_MARGIN_FRAMES)
            run_labeltool(data_dir, f"err_{row.status}_{row.frame}", start, end)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rondo", description="Team Rondo Phase 1 pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="aggregate sanity report on the data bundle")
    check.add_argument("--data-dir")
    check.add_argument("--expected-records", type=int)
    check.set_defaults(handler=_check)

    run = sub.add_parser("run", help="build team-rondo.csv")
    run.add_argument("--data-dir")
    run.add_argument("--config", default=str(DEFAULT_CONFIG))
    run.add_argument("--out")
    run.add_argument("--expected-frames", type=int)
    run.add_argument("--validate-rows", type=int)
    caching = run.add_mutually_exclusive_group()
    caching.add_argument(
        "--no-cache", action="store_true", help="build from the source files; no cache read/write"
    )
    caching.add_argument(
        "--refresh-cache", action="store_true", help="build from the source files, rewrite cache"
    )
    run.set_defaults(handler=_run)

    repro = sub.add_parser("verify-repro", help="run twice in fresh processes, compare SHA-256")
    repro.add_argument("--data-dir")
    repro.add_argument("--config", default=str(DEFAULT_CONFIG))
    repro.add_argument("--expected-frames", type=int, help="as for run (small test bundles)")
    repro.add_argument("--validate-rows", type=int, help="as for run (small test bundles)")
    repro.set_defaults(handler=_verify_repro)

    score = sub.add_parser("score", help="official scores on every validation window")
    score.add_argument("csv", help="a full-length frame,label file")
    score.add_argument("--data-dir", help="hand-labelled windows are read from <data-dir>/labels")
    score.add_argument("--windows", nargs="+", help="score only these windows (default: all)")
    score.add_argument("--bootstrap", type=int, default=0, help="bootstrap draws, e.g. 1000")
    score.add_argument(
        "--result", help="also write a Markdown results file, YYYY-MM-DD-name-topic.md"
    )
    score.add_argument("--owner", help="name written in the results file")
    score.add_argument("--notes", default="", help="notes written in the results file")
    score.set_defaults(handler=_score)

    label = sub.add_parser("label", help="open the labelling window on a stretch of real video")
    label.add_argument("--data-dir")
    label.add_argument("--window", required=True, help="name for the output files, e.g. window1")
    label.add_argument("--start", type=int, required=True, help="first frame of the stretch")
    label.add_argument(
        "--end", type=int, required=True, help="last frame of the stretch (inclusive)"
    )
    label.set_defaults(handler=_label)

    errors = sub.add_parser("errors", help="transitions worth a look: missed, spurious, late")
    errors.add_argument("csv", help="a full-length frame,label file")
    errors.add_argument("--data-dir", help="hand-labelled windows are read from <data-dir>/labels")
    errors.add_argument("--window", required=True, help="truth window to compare against")
    errors.add_argument("--play", action="store_true", help="open each one in the labeltool viewer")
    errors.set_defaults(handler=_errors)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
