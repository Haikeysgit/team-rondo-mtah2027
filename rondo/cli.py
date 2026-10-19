"""Command line: aggregates only, never rows."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

import pandas as pd

from rondo import kit
from rondo.io.load import check_bundle, load_match
from rondo.pipeline import run as run_pipeline
from rondo.post import summarize, validate_official, write_submission

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.toml"
LATER = {
    "label": "Task 14",
    "verify-repro": "Task 7",
}


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
    match = load_match(data_dir, expected_frames=args.expected_frames)
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


def _later(command: str):
    def handler(_: argparse.Namespace) -> int:
        print(f"rondo {command} is available once {LATER[command]} is merged")
        return 2

    return handler


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
    run.set_defaults(handler=_run)

    score = sub.add_parser("score", help="official scores on every validation window")
    score.add_argument("csv", help="a full-length frame,label file")
    score.add_argument("--data-dir", help="hand-labelled windows are read from <data-dir>/labels")
    score.add_argument("--windows", nargs="+", help="score only these windows (default: all)")
    score.add_argument("--bootstrap", type=int, default=0, help="bootstrap draws, e.g. 1000")
    score.add_argument("--result", help="also write results/YYYY-MM-DD-name-topic.md")
    score.add_argument("--owner", help="name written in the results file")
    score.add_argument("--notes", default="", help="notes written in the results file")
    score.set_defaults(handler=_score)

    for command, task in LATER.items():
        later = sub.add_parser(command, help=f"available once {task} is merged")
        later.add_argument("rest", nargs="*")
        later.set_defaults(handler=_later(command))
    return parser


def main(argv: list[str] | None = None) -> int:
    args, extra = _parser().parse_known_args(argv)
    if extra and args.command not in LATER:
        _parser().error(f"unrecognised arguments: {' '.join(extra)}")
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
