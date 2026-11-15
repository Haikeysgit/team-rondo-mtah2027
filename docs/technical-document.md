# Team Rondo: who has the ball, frame by frame

More Than A Hack 2027, Phase 1. Technical document for `team-rondo.csv`.

## Team

Team name `team-rondo`. Commit authors in the repository appear as
`<registered name> (<GitHub or usual name>)`.

| Registered name | GitHub | Role |
|---|---|---|
| Imisioluwa | Haikeysgit | software engineer and football analyst |
| John | Kiki544 | data scientist |
| Dominic | Tega-Efe | software engineer and football analyst |
| Muhammad Ahmad | — | tactical analyst |

## 1. Approach

The task is scored on spellF1: a possession only counts if we recover it as one whole spell. A
system that is right on most frames but breaks one possession into three scores badly. So we
treat the match as a sequence and decide each half as a whole, rather than labelling each frame
on its own.

The pipeline has four stages, each a small module with a fixed interface (`docs/CONTRACTS.md`):

1. **Signals** (`rondo/signals/`) turn the tracking and the Smart Tagging into per-frame evidence:
   which team's player is nearest the ball and how far, ball speed, whether the ball is inside the
   pitch, how many players of each team are moving, and the restart windows and team hints from
   the Smart Tagging.
2. **Emission** (`rondo/emissions/rules.py`) turns the evidence into a probability of `H`, `A` and
   `D` for each frame. It is a set of explicit rules with weights in log-odds, combined by a
   softmax: control by a team's player adds weight to that team, a ball out of bounds or a
   stoppage adds weight to `D`, the Smart Tagging hint adds a small lean.
3. **Decoder** (`rondo/decoder/viterbi.py`) finds the most likely label sequence for each half
   with the Viterbi algorithm. Every change of label has a cost, so a flicker of evidence is not
   worth a switch while sustained evidence is, and the switch lands where that evidence starts.
   Leaving a dead ball is cheaper inside a Smart Tagging restart window.
4. **Post-processing** (`rondo/post/`) writes `X` on frames that are not graded, and never on a
   graded frame: there macroF1 counts an `X` as a miss and spellF1 replaces it with the previous
   label, which the decoder already carries. It forces `D` on frames 110,808–113,741 (before the
   second-half kick-off, as the official FAQ specifies) and writes the file, which the official
   validator then checks.

Why rules and a decoder rather than a learned classifier: the rules forbid hand labels of this match
for any purpose, so there is nothing of this match to train on, and a per-frame classifier would
still need smoothing to produce whole possessions; the official baselines show how much that
smoothing decides. Explicit rules let each threshold be stated in football terms and checked on
its own, and the decoder handles the "whole possession" requirement directly. A learned emission
trained on other matches is a possible upgrade (section 7).

## 2. Assets used and dropped

| Asset | Use |
|---|---|
| Tactical tracking (players and ball) | the main evidence for every signal |
| Tracking metadata | team of each tracking slot, pitch size |
| Smart Tagging | restart windows for the decoder (`SET PIECES`), and a team hint from the phase codes. A tag's `Team` is the team *performing* the event, so for defending phases the hint is flipped to the other team |
| Homography and tactical video | not used by the pipeline. Watched only, to understand what counts as possession; our output is never checked against it |
| Broadcast video and tracking | **dropped**: on its own timeline (cuts, replays, ~164k frames), with no mapping to the tactical frames, and a third of its frames have no homography |
| Broadcast commentary audio | **dropped**: the tactical video has no audio, and the broadcast audio is not aligned to it |

The team hint was given a small weight from the start (0.3 in log-odds), a lean rather than a
decision. Automatic checks afterwards (aggregates only, no setting chosen from them): the Smart
Tagging's 59 restart windows all exist, 58% of them contain the ball leaving the pitch in the
tracking, and 46% name a team. Its team hint covers 70% of graded frames and names the right team
on 70% of the official public frames. The team named on a restart agreed with the official
baseline after only 9 of 23 restarts. Sustained control outweighs the restart bonus, so a wrong
tag rarely decides a possession.

## 3. Training and validation data

No model is trained on labels. The emission's weights and the decoder's costs are set by hand
from football rules and checked on other matches.

| Dataset | Source | Labelled by | Size | Used for |
|---|---|---|---|---|
| Metrica sample games 1, 2, 3 | `github.com/metrica-sports/sample-data` (public) | Metrica's annotators (event data) | 3 matches; 1,745 + 1,935 + 3,620 events; 790 possessions in our answer keys | choosing and checking settings |
| Official public window | the organisers' kit | the organisers | 5 minutes, 17 possessions | checking that the pipeline is correct; never to choose a setting |

The sample games' answer keys are built by code from their events, under the same possession
rules as our system (section 5): a spell starts at the new team's first possession event, the
ball is dead from where it leaves the pitch or from a foul until the next set piece, and a spell
won in play counts only with three passes (the official hint: "more than two consecutive
passes"), or after 2 seconds. Where a match's log records carries (game 3), those 2 seconds must
also hold another action of the new team after it wins the ball (a pass, carry or shot); games 1
and 2 log no carries, so there the time alone decides. Without this rule the events would give 403
possessions for game 3 instead of 280, because annotators log every one-touch recovery
(`rondo/othergames/metrica.py`). Only events decide it, never tracking, which is our system's own
sensor.

Hand labels of this match: before the organisers' rule change of Oct 22–23, the rules allowed
small hand-labelled sets for validation, and we labelled two 5-minute windows of this match. When
the rule changed we deleted them, and rolled back the one setting that had been chosen with them.
No part of the submitted system depends on them.

## 4. Missing and noisy tracking

- The ball is missing on about 41% of graded frames (the official DATA.md). On those frames
  there is no control evidence, `H` and `A` stay level, and the decoder carries the current
  possession through. A stoppage is only inferred when players are slow on average **and** the
  ball is still, missing or inside a restart window: slow players alone are not a stoppage.
- Player identities are anonymous tracking slots that swap. The system never follows a player:
  it only asks which *team's* player is nearest the ball. For the same reason, counting passes
  between teammates was left out.
- Ball speed is a centred difference over ±2 frames; an unknown speed counts
  as slow. The ball is out of bounds only beyond 0.5 m outside the lines, to absorb tracking noise.

## 5. Possession rules implemented

From the official rules, our tactical analyst's answers and, for when the ball is in play, the
Laws of the Game (Law 9). The full rules and edge cases are in `docs/POSSESSION.md`; the main
cases have end-to-end tests (`tests/test_scenarios.py`) and stoppages have unit tests.

| Rule | How | Parameter |
|---|---|---|
| Control | a player of the team nearest a slow ball, sustained over a run of the team's own control frames (its passes and short gaps inside the run) | `control_dist_m` 1.5 m, `control_speed_ms` 8 m/s, `min_control_frames` 25 (1 s; 2 s merged possessions on the sample games), `max_control_gap_frames` 75 |
| Touches, blocks, deflections, won headers | give no evidence, so they cannot flip possession | (only sustained runs count) |
| Loose ball | stays with the last team | no evidence, decoder carries it |
| Switch placement | where the new team gained control, not after the seconds that confirmed it | decoder over the whole half |
| Ball over a line | dead from the frame it is wholly over (Law 9), and still dead while tracking has lost it, until it is seen again | `out_weight` 5, 0.5 m margin, `out_carry_weight` 5 |
| Referee stops play (foul, offside, injury) | dead when play stops, read from players slowing with a still or missing ball | `stopped_speed_ms` 1.0, `still_ball_ms` 0.5, `stopped_weight` 4 |
| Restart | dead until the kick or throw, then the restarting team | `dead_exit_cost` 25, `restart_bonus` 6 |
| Change of possession | one cost per switch | `switch_cost` 8 (4 to 16 score within 0.007 of each other on the sample games) |

## 6. Validation

A change is accepted when it improves spellF1 on the other matches with 90% bootstrap intervals
that do not overlap the current version's, or when it fixes a documented failure with a general
rule. `rondo othergame --bootstrap 1000` runs a configuration on all three sample games and
prints, per match and all together, spellF1, its interval (the official scorer's bootstrap over
whole possessions), macroF1 and the possession count. Correctness is covered separately: the
official validator on every output, end-to-end scenario tests on synthetic matches, and a
golden SHA-256 test of the full pipeline on Linux and Windows.

Every version below runs on the final code (`configs/default.toml`,
`configs/experiments/control-25.toml`, `configs/experiments/starting-defaults.toml`).

| Version | spellF1, three sample games | 90% interval | macroF1 | possessions |
|---|---|---|---|---|
| Submitted configuration (control 1 s, ball last seen out stays out) | 0.6579 | 0.6299–0.6837 | 0.8668 | 668 / 790 |
| Control 1 s | 0.5439 | 0.5166–0.5722 | 0.7390 | 678 / 790 |
| Starting defaults (control 2 s) | 0.4904 | 0.4614–0.5197 | 0.7287 | 591 / 790 |
| Official baseline (`baseline_smooth50`) on this match's public window, for scale | 0.3256 | 0.2140–0.4465 | 0.6207 | 26 / 17 |

## 7. Limitations

- With the starting defaults (control after 2 s) the system merged short possessions. The
  submitted configuration finds 668 possessions on the sample games against the answer keys' 790,
  so it still finds fewer.
- Games 1 and 2 log no carries, so their keys cannot tell a lone touch lost straight back from a
  short dribble then lost, and count both after 2 seconds: those two keys are probably generous on
  such spells. Game 3, whose log can tell, drops 8 of them.
- Counting a ball last seen out as still out gains far less on this match than on the sample games:
  `D` rises by 3 points of the graded frames here against 10 to 14 there, so the sample-game score
  should not be read as a forecast for this match.
- The nearest player is farther from the ball in this match's tracking than in the sample games'
  (a median of 2.4 m, against 0.3 m to 1.3 m), while ball coverage and speeds are alike. The
  control test needs a player within 1.5 m, so it likely fires less often here than on the sample
  games. This is an automatic check of the data, not a reason for a setting: `control_dist_m`
  stays as the sample games decide.
- Dead-ball timing: a stoppage is read from players slowing down, which lags the whistle by
  a second or more, and during long stoppages (a video review, a penalty being set up) walking
  players and a ball moved about can look like play. A "stay dead until a real restart" rule
  cannot be judged on the sample games, which have no Smart Tagging, so it was not adopted.
- Smart Tagging's restart team is unreliable (section 2). Its value cannot be measured on the
  sample games.
- Answer keys from events approximate the official definition of possession; the sample games
  are three matches of other teams.
- With more time: an emission learned from the sample games' answer keys; stable player
  identities, to count passes; and the referee's whistle from the broadcast audio, once aligned to
  the tactical frames.

## 8. Reproduction

```bash
git clone --recurse-submodules https://github.com/Haikeysgit/team-rondo-mtah2027.git
cd team-rondo-mtah2027
uv sync --frozen
# put the official data bundle in ./data, with the kit's layout (tactical/, broadcast/)
uv run rondo run --out team-rondo.csv
```

- Tag of the submitted commit: `submission-2026-11-14`.
- SHA-256 of `team-rondo.csv`:
  `57eba9b4e5e2d68a8ce1de9c6fc92972a214af5c48ea60654dd86f84ef096a04`. `uv run rondo verify-repro`
  runs the pipeline twice in fresh processes and compares the two hashes.
- Run time on a laptop (Intel i7-8650U, 16 GB): about 30 seconds for the full match, with no
  cache.
- Python 3.11 with `uv`; the official kit is a pinned submodule (`vendor/mtah-kit`), used
  unchanged.
