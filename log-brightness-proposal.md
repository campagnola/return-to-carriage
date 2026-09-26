# Proposal: Log messages fade with age and salience

**Status:** implemented (see `git diff`); constants still to be tuned by eye.

## Summary

Each console message is drawn at a brightness in **[0.2, 1.0]** that depends on
how many **turns** ago it was written and on its **salience** (0–1). Salient
messages start brighter and fade more slowly; chatter starts dim and sinks to
the floor within a few turns. Brightness 1.0 is exactly today's console text
colour, so a fully bright line looks the same as it does now.

## Behaviour

```
b0(s)  = B_LOW + (B_HIGH - B_LOW) * s          initial brightness, linear in salience
tau(s) = T_MIN * (T_MAX / T_MIN) ** s          decay constant in turns, geometric in salience
b(age, s) = clamp(b0(s) * exp(-age / tau(s)), 0.2, 1.0)
```

Starting constants (all tunable, all in one place in `hud.py`):

| constant | value | why |
|---|---|---|
| `B_LOW`, `B_HIGH` | 0.5, 1.5 | the raw value may leave [0.2, 1.0]; the clamp makes high-salience lines *hold* at full brightness before they fade, instead of fading from the first turn |
| `T_MIN`, `T_MAX` | 4, 64 turns | geometric, so equal salience steps give equal *ratios* of lifetime |
| floor / ceiling | 0.2 / 1.0 | as specified |

Resulting brightness (computed from the formula above):

| salience | b0 | tau | age 0 | 1 | 2 | 5 | 10 | 20 | 40 | 80 | at floor after |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.50 | 4.0 | 0.50 | 0.39 | 0.30 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 3.7 turns |
| 0.25 | 0.75 | 8.0 | 0.75 | 0.66 | 0.58 | 0.40 | 0.21 | 0.20 | 0.20 | 0.20 | 10.6 |
| 0.50 | 1.00 | 16.0 | 1.00 | 0.94 | 0.88 | 0.73 | 0.54 | 0.29 | 0.20 | 0.20 | 25.8 |
| 0.75 | 1.25 | 32.0 | 1.00 | 1.00 | 1.00 | 1.00 | 0.91 | 0.67 | 0.36 | 0.20 | 58.6 |
| 1.00 | 1.50 | 64.0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.80 | 0.43 | 129 |

Held at full brightness before decay begins: 0 turns for s ≤ 0.5, ~7 turns for
s = 0.75, ~26 turns for s = 1.

The upper clamp is only live because `B_HIGH > 1`. If `b0` were capped at 1 the
ceiling would be inert; I prefer the plateau, but it is a one-constant change.

## Design

### Data: a log entry has text, salience, and birth turn

`MessageLog.lines` is currently `list[str]`. Salience and birth turn are
properties of each *message*, so they belong on a per-line record rather than in
parallel lists that `write` / `set_last_line` / `remove_last_line` would all have
to keep in step:

```python
LogEntry = namedtuple('LogEntry', 'text salience turn')

class MessageLog:
    entries: list[LogEntry]
    turn: int = 0                       # turns elapsed; the age reference
    def write(self, text, salience=DEFAULT_SALIENCE): ...   # stamps turn=self.turn, one entry per line
    def set_last_line(self, line): ...  # keeps the entry's salience and turn
    def advance_turn(self): ...         # turn += 1; version bump; changed()
    @property
    def lines(self): return [e.text for e in self.entries]  # read-only snapshot
```

- **The log owns the turn counter.** `CommandInputHandler` is constructed with
  only the `MessageLog` (`return_to_carriage.py:68`, and tests build it with a
  bare log), so the log has to be able to stamp entries without reaching for the
  scene. Age is "turns elapsed since I was written", which is the log's business.
  Alternative: a `scene.turn` counter with the log holding a reference to it —
  more principled if hunger/cooldowns-in-turns appear later, but it needs a
  second observable and threads a dependency through `CommandInputHandler`.
  Easy to migrate later since only `advance_turn` and the stamping line change.
- `advance_turn()` goes through `_changed()`, so `ConsoleWidget`'s existing
  `log.changed` subscription repaints on every turn with no new wiring.
- `Scene.write(message, salience=DEFAULT_SALIENCE)` passes salience through.
- **Default salience 0.5** (b0 = 1.0, tau = 16): an unannotated message looks as
  it does today for the first turn, then fades gently. Every existing call site
  keeps working unchanged.

### Turn boundary

`DungeonMaster.end_turn()` is the only turn boundary today (called once, from
`move_player`), so it gains one line: `self.scene.log.advance_turn()`.
Consequence to be aware of: commands like `take`/`drop`/`read` do not currently
end a turn, so messages only age when the player *moves*. That matches "turns"
as the game defines them; if commands should cost turns that is a separate
change and this feature picks it up for free.

### Rendering: display policy stays in the HUD

The log stores facts (salience, birth turn); the mapping to a colour is
presentation, so it lives in `hud.py` as a pure function:

```python
def brightness(age, salience): ...   # the formula above, constants at module top
```

`ConsoleWidget.repaint` walks `log.entries[-rows:]`, wraps each, and writes every
wrapped chunk with `fg = (R*k, G*k, B*k, A)` where `k = brightness(log.turn - entry.turn, entry.salience)`
and `(R, G, B, A) = FG`. Scaling RGB while leaving alpha alone means brightness
1.0 reproduces today's `FG` exactly and dimming reads as "toward the console
background" rather than "the map shows through the text".

### Command prompt lines

`CommandInputHandler` uses the log as an editable last line (`> git_`). It must
not fade or dim, so its `write` calls pass `salience=1.0` (b0 = 1.5, pinned at
1.0). `set_last_line` preserves the entry, so the prompt stays bright while
typed. The command echo written by the interpreter (`"\n> take"`) is an ordinary
message at the default salience.

## Files touched

| file | change |
|---|---|
| `carriage_return/scene.py` | `LogEntry`; `MessageLog` stores entries, `turn`, `advance_turn()`, `lines` property; `Scene.write(..., salience=)` |
| `carriage_return/dm.py` | `end_turn()` calls `scene.log.advance_turn()` |
| `carriage_return/hud.py` | `brightness()` + constants; `ConsoleWidget.repaint` colours per line |
| `carriage_return/input.py` | prompt `write` passes `salience=1.0` |
| `ARCHITECTURE.md` | one paragraph in the HUD section (and its `ConsolePainter` wording is already stale vs `ConsoleWidget`) |

No rendering-backend change: the grid already carries per-cell `fgcolor`, and
`Widget.write(..., fg=)` already sets it.

## Tests (headless, no windows)

- `tests/test_log.py`: entries carry salience and the current turn; a multi-line
  write stamps every line; `set_last_line` keeps salience/turn;
  `advance_turn` bumps `version` and fires `changed`; `lines` still returns
  plain strings.
- `tests/test_hud.py`: `brightness()` — bounds (never < 0.2 or > 1.0 over a wide
  age/salience grid), monotone non-increasing in age, monotone non-decreasing in
  salience at fixed age, plateau at 1.0 for high salience; console cell `fgcolor`
  for a fresh vs aged line; wrapped chunks of one entry share a colour; a turn
  advance repaints the console.
- Existing tests that read `scene.log.lines` keep working, with one exception:
  the `messages` fixture in `tests/test_actions.py` returns the *live* list and
  relies on later appends showing up in it. A property returns a snapshot, so
  that fixture needs to return a live view (e.g. yield the log and have tests
  read `.lines` at assertion time).
- Visual check with the existing screenshot harness (`agent_helpers/`) to judge
  legibility of the 0.2 floor against a bright map: RGB 0.2 at alpha 0.5 over the
  0.4-alpha console background may be too faint to read, in which case the fix
  is raising the floor or the base `FG` alpha, not the formula.

## Open questions

1. **`Percept.salience` is a different scale.** `perception.py` defines salience
   as unbounded and log-scaled (0 = "maximally cared about", negatives matter
   less; the sword uses 1.0, the sewer warning 10.0), and it only decides
   auto-log vs held. The 0–1 log salience here is a separate quantity.
   Proposal: leave percepts on the default salience for now, and decide the
   mapping (e.g. a squash of percept salience into 0–1) when there is a
   percept whose fade actually matters. Overloading the name without a mapping
   would silently misbehave (`salience=10.0` is out of range for the log).
   Should out-of-range log salience raise? I would say yes — validate in
   `MessageLog.write` rather than clamp, per the no-silent-fixups rule.
2. **Constants.** The `T_MIN`/`T_MAX` lifetimes (about 4 to 130 turns to floor)
   are a guess at how long a move-based turn feels; worth tuning by hand.

## Out of scope

Brightness for anything other than the console; per-line colour/hue by message
type; changing what counts as a turn; mapping percept salience onto log salience.
