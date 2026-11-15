# team-rondo: possession labels for More Than A Hack 2027, Phase 1

This repository is Team Rondo's (`team-rondo`) submission to Phase 1 of More Than A Hack 2027.
For every frame of the tactical video of FC Barcelona against Athletic Club it writes who has
the ball, `H` (home), `A` (away), `D` (dead ball) or `X` (not graded), into `team-rondo.csv`.
Per-frame evidence from the tracking and the Smart Tagging events becomes per-frame label
probabilities, and a Viterbi decoder with a cost per possession change turns them into whole
possessions. The approach, the possession rules and the limitations are in
[docs/technical-document.md](docs/technical-document.md).

The match data is not in this repository and never was: it is licensed for the competition only.

## Team

Commit authors appear as `<registered name> (<GitHub or usual name>)`.

| Registered name | GitHub | Role |
|---|---|---|
| Imisioluwa | [Haikeysgit](https://github.com/Haikeysgit) | software engineer and football analyst |
| John | [Kiki544](https://github.com/Kiki544) | data scientist |
| Dominic | [Tega-Efe](https://github.com/Tega-Efe) | software engineer and football analyst |
| Muhammad Ahmad | — | tactical analyst |

## Requirements

- [uv](https://docs.astral.sh/uv/) (installs Python 3.11 and every pinned dependency from `uv.lock`)
- [Git](https://git-scm.com/)
- Windows, macOS or Linux. The tests run on Linux and Windows on every change.

## Install

```bash
git clone --recurse-submodules https://github.com/Haikeysgit/team-rondo-mtah2027.git
cd team-rondo-mtah2027
uv sync --frozen
```

`--recurse-submodules` fetches the organisers' kit (`vendor/mtah-kit`, pinned to one commit),
whose validator, scorer and match constants are used unchanged.

## Data

Unpack the match bundle into `./data`, in the kit's layout:

```
data/
  tactical/
    BAR-ATH_tactical_atd.xml
    BAR-ATH_tactical_atd.txt
    BAR-ATH_tactical_homography.csv
    BAR-ATH_tactical_smart_tagging.xml
    BAR-ATH_tactical.mp4
```

Or keep it anywhere and set `MTAH_DATA_DIR` to the folder that contains `tactical/` (as an
environment variable, or in a `.env` file at the repository root; see `.env.example`).
`data/` is ignored by Git. A run caches the parsed tracking in `<data folder>/cache/`; the
cache is tied to the source files and the code and library versions, and
`uv run rondo run --no-cache --out team-rondo.csv` ignores it altogether.

## Reproduce the submission

One command:

```bash
uv run rondo run --out team-rondo.csv
```

It prints aggregates only (label counts, number of possessions, the file's SHA-256) and runs the
official validator on the file. To run the validator yourself:

```bash
uv run python vendor/mtah-kit/evaluation/validate_submission.py team-rondo.csv
```

There is no training step: every tuned value is in `configs/default.toml`. The same data gives a
byte-identical file on every run and platform; `uv run rondo verify-repro` runs the command twice
in fresh processes, first from the source files and then from the cache, and compares the two
files' SHA-256.

- Run time: about 30 seconds for the full match on a laptop (Intel i7-8650U, 16 GB), with no
  cache.

## Tests

```bash
uv run pytest
```

The tests use synthetic matches generated in code, so they need no data.

## Validation on other matches

Settings are chosen on other matches, never on the competition match. We use the three
anonymised matches of Metrica Sports' public sample data (tracking and event data), with answer
keys built by code from their events (`rondo/othergames/metrica.py`):

> Metrica Sports, *Sample tracking and event data*,
> <https://github.com/metrica-sports/sample-data>

```bash
git clone --depth 1 https://github.com/metrica-sports/sample-data ~/Documents/mtah-other-games
uv run rondo othergame --bootstrap 1000
```

## Submitted version

- Tag of the submitted commit: `submission-2026-11-14`
- SHA-256 of the submitted `team-rondo.csv`:
  `57eba9b4e5e2d68a8ce1de9c6fc92972a214af5c48ea60654dd86f84ef096a04`

## Licence

[MIT](LICENSE). The licence covers this repository's code and documents only, not the match
data. The kit in `vendor/mtah-kit` keeps its own licence.
