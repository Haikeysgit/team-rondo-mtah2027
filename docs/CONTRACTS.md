# Contracts

The formats modules pass to each other. Code against these, never against another module's
internals. `rondo/contracts.py` holds the code version and `validate_*` checks that the tests
run on every module's output.

Changing a contract: tell the team first, then record the decision and its reason.

All tables are pandas DataFrames. Frames are **video frames** `0 … 199318`.

## C1 — Match data (output of `rondo.io`)

`MatchData(frames, players, tags, meta)`

**`meta`** — `Meta(n_frames=199319, fps=25, pitch=(105.3, 68.0), half_blocks=[(1, 11185, 90651), (2, 110808, 190197)], home_name, away_name)`

**`frames`** — one row per video frame

| column | type | notes |
|---|---|---|
| `frame` | int64 | 0 … n-1, unique, sorted |
| `half` | Int8 | 1, 2, or NA outside the halves |
| `graded` | bool | inside a half block |
| `ball_x`, `ball_y` | float64 | 0–1, NaN when missing |

The data has no ball-status or owning-team field.

**`players`** — one row per tracked object per frame

| column | type | notes |
|---|---|---|
| `frame` | int64 | |
| `slot` | int16 | track slot, not a person; slots swap identity |
| `team` | str | `H` or `A` |
| `x`, `y` | float64 | 0–1 |
| `speed` | float64 | m/s |

**`tags`** — Smart Tagging

| column | type | notes |
|---|---|---|
| `start_frame` | int64 | |
| `end_frame` | int64 | exclusive |
| `code` | str | upper case, e.g. `SET PIECES` |
| `team` | str or null | `H`, `A`, or null (`N/A` in the data). Null is NaN under pandas 3: test with `pd.isna`, not `is None` |
| `type` | str or null | e.g. `Throw-in`, `Corner Kick`. Null is NaN: test with `pd.isna` |
| `half` | Int8 | |

## C2 — Signal

`fn(match: MatchData) -> DataFrame[frame, …columns]`, registered with
`@register_signal("name")`. One row per frame. Column names unique across all signals. The pipeline calls `fn(match)`,
runs `validate_signals`, and merges all signals on `frame` in config order (a column-name clash
raises `ValueError` naming both signals).

## C3 — Emission model

```python
fit(signals, labels) -> self
predict(signals) -> DataFrame[frame, p_H, p_A, p_D]   # each row sums to 1
```

Registered with `@register_emission("name")`. The pipeline builds it as
`Cls(**config.get("emission_params", {}))`, calls `.fit(signals, labels)` only when
`pipeline.run(..., labels=...)` is given, then `.predict(signals)`.

## C4 — Decoder

`decode(emissions, tags, meta, params) -> DataFrame[frame, label]` with labels `H`, `A`, `D`,
registered with `@register_decoder("name")`. The pipeline passes
`params = config.get("decoder_params", {})`; each decoder builds its own typed params from the
dict.

## C5 — Submission

The official format: header `frame,label`, 199,319 rows, frames `0 … 199318` in order, labels
`H` `A` `D` `X`, `\n` line endings. `rondo.post.write_submission` must pass
`vendor/mtah-kit/evaluation/validate_submission.py`.

## Using `tests/synthetic.make_match`

`make_match(steps, n_frames=3000, start=100, seed=0)` plays a script of `Step(event, team, seconds)`
and returns `(MatchData, truth)`; the module docstring describes each event.

- Order matters. After `out` only `restart`, `dribble`, `missing` or `out` are allowed; `pass`,
  `deflection`, `aerial` and `interception` raise `ValueError`. `interception` needs the other
  team to hold the ball and cannot be first. Its flight takes half the step, so keep it to about
  4 s at most. `team` is ignored by `missing`.
- A deflection touch is usually one frame. Players are anonymous slots (0-10 `H`, 11-21 `A`); the
  taker of a restart is the player of that team nearest a line, not a fixed slot.
- Truth switches at fixed frames (interception at `n // 2`, `D` from `max(1, n // 4)` of an `out`
  step), never depending on the seed. Frames outside the half block have no players or ball and
  truth `X`.
- A null tag team is NaN. To test a restart whose tag is `N/A`, edit the result:
  `m.tags["team"] = pd.Series([pd.NA] * len(m.tags), dtype="str")` (or `m.tags.loc[:, "team"] = np.nan`).
- Second-half scenario: build the match, then change `m.meta` with `dataclasses.replace(m.meta,
  half_blocks=[(2, start, end)])` and set `frames["half"]` and the tags' `half` to 2 to match.
