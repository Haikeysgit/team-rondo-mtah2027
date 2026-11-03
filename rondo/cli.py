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


def _tags(args: argparse.Namespace) -> int:
    from rondo import config as runtime_config
    from rondo.eval import load_windows, read_labels
    from rondo.eval.tagcheck import tag_report

    data_dir = _data_dir(args.data_dir) or runtime_config.data_dir()
    match = load_match(data_dir, expected_frames=args.expected_frames)
    for key, value in tag_report(match, None).items():
        if "vs_truth" not in key:  # per window, below
            print(f"{key}: {'n/a' if value is None else value}")
    if args.against:  # a stand-in judge, e.g. the official baseline, which never reads the tags
        path = Path(args.against)
        references = {path.name: read_labels(path, n_frames=match.meta.n_frames)}
    else:
        references = load_windows(data_dir)
    for name, truth in references.items():
        if len(truth) != match.meta.n_frames:
            print(f"{name}: skipped ({len(truth)} frames, the match has {match.meta.n_frames})")
            continue
        report = tag_report(match, truth)
        print(
            f"{name}: restart team right {report['restart_team_vs_truth_pct']}% "
            f"of {report['restart_team_vs_truth_n']}, "
            f"team hint right {report['prior_vs_truth_pct']}%"
        )
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


def _error_breakdown(root: Path, game: int, pred: pd.Series, truth: pd.Series) -> pd.DataFrame:
    """``error_moments(pred, truth)`` with an ``event`` column: the public event log's own name
    for what was happening at each missed, spurious or late frame."""
    from rondo.othergames.metrica import load_events, nearest_event_kind
    from rondo.viz.errors import error_moments

    events = load_events(root, game)
    out = error_moments(pred, truth)
    out["event"] = [nearest_event_kind(events, frame) for frame in out["frame"]]
    return out


def _print_error_breakdown(label: str, moments: pd.DataFrame) -> None:
    """Counts only: how many of each status, and which event kind they cluster
    around, never the frames themselves."""
    totals = moments["status"].value_counts()
    summary = ", ".join(f"{status} {int(totals.get(status, 0))}" for status in totals.index)
    print(f"{label} errors ({summary or 'none'}):")
    grouped = moments.groupby(["status", "event"], observed=True).size()
    counts = grouped.sort_values(ascending=False)
    for (status, event), n in counts.items():
        print(f"  {status:<9} {event:<14} {n}")


def _quiet_ball_check(root: Path, game: int, match, pred: pd.Series, truth: pd.Series) -> dict:
    """Of the spurious switches more than 2 s from any public event, how many follow a quiet
    ball: median ball speed under 0.5 m/s (``still_ball_ms``) in the second before the switch,
    rather than a pass or carry actually arriving there.

    Checks a specific guess at what is behind that cluster: a team holds the ball through a quiet
    spell while an opponent happens to stand within control_dist_m of it for over a second, which
    reads as that opponent's control even though nothing was actually won.
    """
    from rondo.othergames.metrica import load_events, nearest_event_kind
    from rondo.registry import SIGNALS
    from rondo.viz.errors import error_moments

    speed = SIGNALS["ball_motion"](match).set_index("frame")["ball_speed_ms"]

    # Baseline: the same "fully untracked second before" rate, over every frame of the match, not
    # just the error ones. Control evidence cannot accumulate while the ball is missing, so any
    # switch tends to land shortly after a gap ends; this says whether the error frames land there
    # more than a random frame would, or whether that is just how much of the match looks. Computed
    # unconditionally (not only when there are spurious switches) so a game that happens to have
    # none still contributes its full share of frames to the combined baseline.
    tracked_in_window = speed.notna().astype(int).rolling(25, min_periods=1).sum().shift(1)
    fully_untracked = tracked_in_window.fillna(0) == 0
    baseline = {
        "baseline_frames": int(fully_untracked.count()),
        "baseline_untracked_frames": int(fully_untracked.sum()),
    }

    moments = error_moments(pred, truth)
    spurious = moments[moments["status"] == "spurious"]
    if spurious.empty:
        return {"n": 0, "quiet": 0, "active": 0, "untracked": 0, **baseline}

    events = load_events(root, game)
    mask = pd.Series(
        [nearest_event_kind(events, frame, max_gap=50) == "none" for frame in spurious["frame"]],
        index=spurious.index,
    )
    far = spurious[mask]

    quiet = active = untracked = 0
    for frame in far["frame"]:
        window = speed.loc[max(0, frame - 25) : frame - 1].dropna()
        if window.empty:
            untracked += 1
        elif window.median() < 0.5:
            quiet += 1
        else:
            active += 1

    return {"n": len(far), "quiet": quiet, "active": active, "untracked": untracked, **baseline}


def _print_quiet_ball_check(label: str, stats: dict) -> None:
    print(f"{label} quiet-ball check ({stats['n']} spurious switches more than 2s from any event):")
    if stats["n"] == 0:
        return
    baseline_pct = 100 * stats["baseline_untracked_frames"] / stats["baseline_frames"]
    print(f"  quiet before switch      {stats['quiet']}")
    print(f"  active before switch     {stats['active']}")
    print(f"  ball untracked           {stats['untracked']}")
    print(f"  baseline (any frame)     {baseline_pct:.1f}% fully untracked")


def _print_table(table: pd.DataFrame) -> None:
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


def _othergame(args: argparse.Namespace) -> int:
    """Run a config on Metrica's public sample games and score it against their event answer keys.

    One row per game present (``sample1``..``sample3``) and, with more than one, an ``all`` row:
    the games scored together, joined end to end with an ``X`` frame between them so no
    possession runs from one match into the next.
    """
    import os

    from rondo.eval import score_all, write_result
    from rondo.othergames.metrica import GAMES, answer_key, game_folder, load_sample_game

    if args.result and not args.owner:
        _parser().error("--result needs --owner")
    default_root = Path.home() / "Documents" / "mtah-other-games"
    root = Path(args.root or os.environ.get("MTAH_OTHER_GAMES_DIR") or default_root)
    wanted = GAMES if args.game == "all" else (int(args.game),)
    games = [game for game in wanted if game_folder(root, game).is_dir()]
    if not games:
        print(
            f"no sample game under {root}: clone github.com/metrica-sports/sample-data there, "
            "or pass --root"
        )
        return 1
    with open(args.config, "rb") as handle:
        config = tomllib.load(handle)

    tables, preds, truths, moments, quiet = [], [], [], [], []
    for game in games:
        match = load_sample_game(root, game)
        truth = answer_key(root, game, match.meta)
        pred = run_pipeline(match, config).set_index("frame")["label"]
        table = score_all(pred, {f"sample{game}": truth}, bootstrap=args.bootstrap)
        tables.append(table[table["window"] == f"sample{game}"])
        preds.append(pred)
        truths.append(truth)
        if args.errors:
            moments.append((game, _error_breakdown(root, game, pred, truth)))
        if args.quiet_ball:
            quiet.append((game, _quiet_ball_check(root, game, match, pred, truth)))
    if len(games) > 1:
        gap = pd.Series(["X"])
        pred = pd.concat([part for p in preds for part in (p, gap)], ignore_index=True)
        truth = pd.concat([part for t in truths for part in (t, gap)], ignore_index=True)
        pred.index = truth.index = pd.RangeIndex(len(truth), name="frame")
        table = score_all(pred, {"all": truth}, bootstrap=args.bootstrap)
        tables.append(table[table["window"] == "all"])
    table = pd.concat(tables, ignore_index=True)
    _print_table(table)
    if args.errors:
        for game, breakdown in moments:
            _print_error_breakdown(f"sample{game}", breakdown)
        if len(moments) > 1:
            combined = pd.concat([breakdown for _, breakdown in moments], ignore_index=True)
            _print_error_breakdown("all", combined)
    if args.quiet_ball:
        for game, stats in quiet:
            _print_quiet_ball_check(f"sample{game}", stats)
        if len(quiet) > 1:
            keys = (
                "n",
                "quiet",
                "active",
                "untracked",
                "baseline_frames",
                "baseline_untracked_frames",
            )
            total = {key: sum(stats[key] for _, stats in quiet) for key in keys}
            _print_quiet_ball_check("all", total)
    if args.result:
        command = (
            f"rondo othergame --game {args.game} --config {args.config} "
            f"--bootstrap {args.bootstrap}"
        )
        write_result(Path(args.result), args.owner, command, table, args.notes)
        print(f"wrote {args.result}")
    return 0


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
    _print_table(table)
    if args.result:
        command = f"rondo score {args.csv} --bootstrap {args.bootstrap}"
        if args.windows:
            command += f" --windows {' '.join(args.windows)}"
        write_result(Path(args.result), args.owner, command, table, args.notes)
        print(f"wrote {args.result}")
    return 0


def _rebuild(args: argparse.Namespace) -> int:
    """Rewrite ``labels/<window>.csv`` from its marks file, e.g. after fixing a mark by hand."""
    from rondo import config as runtime_config
    from rondo.labeltool.expand import expand_transitions

    data_dir = _data_dir(args.data_dir) or runtime_config.data_dir()
    marks_path = data_dir / "labels" / "marks" / f"{args.window}_marks.csv"
    try:
        marks = pd.read_csv(marks_path).sort_values("frame", kind="stable")
        labels = expand_transitions(marks, args.start, args.end)
    except (OSError, ValueError, KeyError) as error:
        print(f"not rebuilt: {marks_path}: {error}")
        return 1
    labels.to_csv(data_dir / "labels" / f"{args.window}.csv")
    print(f"rebuilt {args.window}.csv from {len(marks)} mark(s)")
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
    print("frame-level output from the match data: read it, do not paste it into an AI chat")
    for status in ("missed", "spurious", "late"):
        print(f"{status}: {int((moments['status'] == status).sum())}")

    top = moments.head(20)
    for row in top.itertuples(index=False):
        # a spurious row has no truth side; missing values read as NaN once rows are mixed
        from_, to = (
            (row.pred_from, row.pred_to) if pd.isna(row.gt_from) else (row.gt_from, row.gt_to)
        )
        change = f"{from_} -> {to}"
        print(f"{format_timestamp(row.frame)}  frame {row.frame:>7}  {row.status:<8} {change}")

    if args.play:
        from rondo.labeltool import app

        n_frames = len(truth)
        session = app.open_session(data_dir)  # the video is opened and counted once for all
        try:
            for row in top.itertuples(index=False):
                start = max(0, row.frame - PLAY_MARGIN_FRAMES)
                end = min(n_frames - 1, row.frame + PLAY_MARGIN_FRAMES)
                window = f"err_{row.status}_{row.frame}"
                app.run(data_dir, window, start, end, session=session, read_only=True)
        finally:
            session.close()
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rondo", description="Team Rondo Phase 1 pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="aggregate sanity report on the data bundle")
    check.add_argument("--data-dir")
    check.add_argument("--expected-records", type=int)
    check.set_defaults(handler=_check)

    tags = sub.add_parser("tags", help="how well Smart Tagging lines up with tracking and truth")
    tags.add_argument("--data-dir")
    tags.add_argument("--expected-frames", type=int)
    tags.add_argument("--against", help="compare with this full-length frame,label file instead")
    tags.set_defaults(handler=_tags)

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

    other = sub.add_parser("othergame", help="score a config on Metrica's public sample games")
    other.add_argument("--root", help="checkout of github.com/metrica-sports/sample-data")
    other.add_argument("--game", choices=["1", "2", "3", "all"], default="all")
    other.add_argument("--config", default=str(DEFAULT_CONFIG))
    other.add_argument("--bootstrap", type=int, default=0, help="bootstrap draws, e.g. 1000")
    other.add_argument("--result", help="also write a Markdown results file")
    other.add_argument("--owner", help="name written in the results file")
    other.add_argument("--notes", default="", help="notes written in the results file")
    other.add_argument(
        "--errors",
        action="store_true",
        help="also break missed/spurious/late counts down by the nearest event kind",
    )
    other.add_argument(
        "--quiet-ball",
        action="store_true",
        dest="quiet_ball",
        help="of the spurious switches far from any event, how many follow a quiet ball",
    )
    other.set_defaults(handler=_othergame)

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

    rebuild = sub.add_parser("rebuild", help="rewrite a labelled window from its marks file")
    rebuild.add_argument("--data-dir")
    rebuild.add_argument("--window", required=True, help="window name, e.g. w1")
    rebuild.add_argument("--start", type=int, required=True, help="first frame of the window")
    rebuild.add_argument("--end", type=int, required=True, help="last frame of the window")
    rebuild.set_defaults(handler=_rebuild)

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
