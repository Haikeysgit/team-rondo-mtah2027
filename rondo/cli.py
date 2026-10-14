"""Command line: aggregates only, never rows."""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

from rondo import kit
from rondo.io.load import check_bundle, load_match
from rondo.pipeline import run as run_pipeline
from rondo.post import summarize, validate_official, write_submission

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.toml"
LATER = {
    "score": "Task 8",
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
