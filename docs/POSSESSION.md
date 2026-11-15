# Possession rules

The main cases have end-to-end scenario tests in `tests/test_scenarios.py`, and stoppages have
unit tests in `tests/emissions/test_rules.py`. Advantage, goalkeeper catches and punches, and
dropped balls follow from the same rules but have no test of their own.

Where the official rules are silent we follow our tactical analyst's answers, and for when the
ball is in or out of play, the Laws of the Game (IFAB, Law 9). The public 5 minutes of official
answers show how the organisers apply these rules; they check that the pipeline runs correctly
and never choose a setting.

## Official rules (summary of CHALLENGE.md)

- Ball in play: one team always has possession. No neutral state.
- Loose ball (50/50, ball running free after a tackle): the last team keeps it.
- Possession changes when the other team has the ball **under control**.
- Not a change: a single touch without control, a deflected pass, a won header without control.
- Every restart (kickoff, throw-in, corner, goal kick, free kick) has a clear owner.
- Dead ball is `D`, never a possession, even if a team will restart.
- The official answers come from corrected tracking: `D` where it marks the ball dead,
  otherwise the team it names as owner.
- Official control proxies (CHALLENGE.md, Oct 15): a player **keeps the ball for a few seconds**,
  or a team **completes more than two consecutive passes**. These are suggestions, not the
  specification.

## Our control criteria

A team has control when it keeps the ball for a sustained spell, its own passes included. Touches
don't count: a ball kept for a second is control, three touches lost straight back are not. Our
tactical analyst and the official hint agree on the principle; the length of the spell was
chosen on other matches.

In the pipeline (`rondo/emissions/rules.py`, `RulesParams`; submitted values from
`configs/default.toml`):

| Rule | Parameter | Submitted |
|---|---|---|
| A player is on the ball when nearest to it within | `control_dist_m` | 1.5 m |
| ... and the ball moves slower than (a pass in flight is nobody's) | `control_speed_ms` | 8 m/s |
| A team's run of control lasts at least | `min_control_frames` | 25 frames (1 s; the code's default is 2 s, which merged possessions on the sample games) |
| Its own passes and tracking gaps inside one run, up to | `max_control_gap_frames` | 75 frames (3 s) |
| A run is broken by | control by the other team | |

- Only sustained runs give evidence. A touch, block or deflection gives none, so it cannot flip
  possession, whatever the decoder's `switch_cost`.
- The switch is placed where the new team **gained** control, not after the second that confirmed
  it: the run is judged on the whole match, and its evidence starts at its first frame.
- A team's passes between teammates stay inside its run, so a passing move forms a sustained run.
  Counting passes directly needs player identities, which the tracking swaps between slots; it is
  left out.
- Tune `min_control_frames` and `control_dist_m` on matches other than this one, never against the
  text alone and never with hand labels of this match.

## Restarts

- A restart (kick-off, throw-in, corner, goal kick, free kick) belongs to the restarting team.
- The ball is **dead until it is kicked or thrown**. The restart team's possession starts at that
  frame, not when the players line up or the referee signals.
- The Smart Tagging restart window (`tag_restart_window`, about 5 s around the restart) is
  evidence of a stoppage and names the restarting team; the decoder anchors the restart there.
- A throw-in tagged `N/A` gets its team from the tracking, not the tag.
- **Dropped ball** (Law 8): dead until the ball touches the ground, then it belongs to the team
  that last touched it (the goalkeeper's team when dropped in the penalty area).

## Dead ball

When the ball goes dead follows Law 9:

- **Ball out of play over a line**: dead the moment it has **wholly crossed** the goal line or
  touchline. No whistle is needed. In the pipeline: `ball_in_bounds` is False (0.5 m margin for
  tracking noise) and the frame is `D`. Tracking usually loses the ball as it leaves the pitch, so
  a ball last seen out of bounds stays dead until it is seen again (`out_carry_weight`).
- **Play stopped by the referee** (foul, offside, injury, any other stoppage): dead **at the
  whistle**. The tracking has no whistle, so we read it from the players: they slow down
  (`team_speed_mean_m_s < stopped_speed_ms`) while the ball is still, missing or inside a restart
  window. Slow players alone are not a stoppage.
- **Advantage played**: the ball stays alive and possession continues.
- **Offside**: alive until the whistle, even though the offence happened at the pass.
- **Goal**: dead from the moment the ball crosses the goal line until the kick-off is taken.
- **Injury stoppage**: dead from the whistle until the restart (often a dropped ball).
- **Frames 110,808–113,741** (before the second-half kick-off): always `D`, forced by
  `rondo.post.finalize`.
- A false short `D` splits a possession in spellF1, so a brief pause inside play (under a second)
  stays with the team in possession.

## Loose balls

The last team in control keeps the ball until the other team gains control (see above):

- Tackle, then the ball runs loose before a teammate collects it: possession changes when the
  teammate **has control**, not at the tackle.
- Shot blocked by a defender, rebound: the attacking team keeps it unless the defenders gain
  control.
- Shot off the post back into play: the attacking team keeps it until the defenders gain control.
- High clearance landing with nobody near: the team that had the ball before the clearance keeps
  it until the other team gains control.
- Aerial duel (goal kick, long ball): a won header is not control. The team that kicked keeps it
  until the opponents gain control.
- Goalkeeper punch: the ball in the air still belongs to the attacking team.
- Goalkeeper catch: control at once, the keeper's team has the ball from the catch.

## Edge cases

| Situation | Expected label |
|---|---|
| Pass between teammates | unchanged |
| Interception, then 2 s or more on the ball | switches to the intercepting team, from the interception |
| Single touch, block or deflection by the other team | unchanged |
| Header won, no control | unchanged |
| Ball wholly over a line | `D` from that frame |
| Foul, offside or injury | `D` from the whistle (players stopping) |
| Advantage played | unchanged, not `D` |
| Restart | `D` until the kick or throw, then the restarting team |
| Ball missing in tracking, inside a possession | unchanged |
| Throw-in tagged `N/A` | decided by tracking, not by the tag |
| Keeper catches the ball | the keeper's team, from the catch |
| Keeper punches the ball | unchanged while it is in the air |
| Goal | `D` until the kick-off |
| Frames 110,808–113,741 (before the second-half kick-off) | `D` |
| Players slow down but the ball is still being passed | unchanged, not `D` |
| Brief stop inside a possession (under a second) | unchanged, not `D` (a false short `D` splits the possession in spellF1) |

## Disagreements with the public answers

None logged. Since the organisers' rule changes, the 5 public minutes are used only to check
that the pipeline runs correctly, never to choose a setting.
