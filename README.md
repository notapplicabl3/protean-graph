# PROTEAN

PROTEAN runs Claude Code inside a small loop that gives a thought durable state, a dollar budget, a console for the human thinking with it and a grade for every unit of work.

A session starts from a thought. A thinker seat (one Claude Code session) answers each call with one move that moves the thought forward: `ideas` (options to pick from), `explore` (a contained worker tries a possibility in a clone; its branch comes back unverified and is never integrated), `task` (units with checkable predicates), `ask` (a question for you) or `rest`. For each unit a contained editor subagent works in a clone of the workspace on its own branch. A grader checks the predicates; only the grader marks a unit passed. The brain keeps the thought record (the thought, where it stands, what is open, what you decided, every move) in `task.json` and renders it as `thought.md`. A session never ends: after `ideas` or `rest` it waits for your next thought, and it stops only at a dollar or tick ceiling.

This is a sanitized public mirror of a private working repository; see § What is withheld and why.

## Run it

```sh
uv sync --locked
uv run pytest tests/protean -q   # the battery; a shim on PATH fails any test that reaches the real binary
uv run protean dry think         # scripted seats: ideas → a scripted think picks one → explore → task → rest; no model call
uv run protean dry think-rest    # a rest parks the session (exit 12); `ctl think` + `resume` wakes it
uv run protean dry dry           # a task move to a rest in a scratch root (every scenario now parks on its rest or its ask: exit 12)
uv run protean dry stuck         # the thinker asks a question; exits 12
uv run protean dry deps --serve  # the fixture brain: serves its console on a scratch root, starts paused, waits on holds;
                                 #   `--mode hold`, `--delay S` between scripted calls, `--ticks N` for a ceiling to extend
uv run protean dry deps          # a unit planned first waits for the two it needs, then sees their work
uv run protean dry branch        # a unit renames a file an earlier unit pinned; a reader admits the hand-over
uv run protean dry weaken        # a unit weakens an accepted test; a reader catches it, the retry is told
```

| Verb | What it does |
|---|---|
| `protean think "<thought>" [--workspace <path>] [--project <slug>] [--mode auto\|hold]` | Start a session (`hold`: a task move's units, every retry and every explore wait for `protean ctl approve`). Spends money. Use a git clone as the workspace (none given: the operator profile's default, else a fresh one-commit repo under `brain/protean/workspaces/<session>/`); the editor's work lands there as branches `unit/<session>/<unit>` and `explore/<session>/<n>`, and each passed unit moves `task/<session>` forward. Nothing is merged; `main` is never touched. A new `think` while another session is parked is allowed; a live brain refuses a second. |
| `protean resume [<session>] [--extend N \| $X] [--abandon]` | Wake the current (or the named) session after a thought or an answer; widen the tick or dollar ceiling of a stopped one; or close it. |
| `protean status` | The thought record (thought, standing, open, decided, moves), tick, terminal, stop reason, units and their statuses, cost, open question. |
| `protean dry <scenario>` | Run a fixture scenario (`think`, `think-rest`, `dry`, `stuck`, `deps`, `deps-broken`, `deps-blocked`, `deps-delivery-only`, `branch`, `weaken`) in a temporary root. |

`--brain PATH` picks the root (default `brain/protean/`). While a task runs, the brain serves a console on 127.0.0.1 (port + token in `brain/protean/state/<task>/console.json`): `protean ctl pause|resume|stop|mode auto\|hold|approve|think <text> [--decide]|answer <id> <text>|extend $X|N|snapshot|watch`. A `think` is your contribution to the thought (the thinker reads it as direction, answer, possibility or decision); `--decide` records it verbatim under `decided`, which nothing else writes. A question, a ceiling or a wait for a thought makes the brain wait for the console; with no client attached for `unattended_seconds` (config, default 30 min) it notifies once and parks `interrupted` — `protean ctl answer` then works offline against `task.json`, and `protean resume` continues. Contract: `docs/console-api.md`. Exit codes: stopped 11, interrupted (parked on a question or a thought) 12, Ctrl-C 130, usage or error 2.

The brain learns who it runs for from one profile: copy `brain/protean/operator.example.json` to `operator.json` (gitignored) and fill in your name, how you want to be written to, and a default workspace. The thinker is told the name and preferences with every call; nothing about the person is written into a prompt, a string or a test.

Caps and each seat's model (`director_model`, `worker_model`, `reader_model`) live in `brain/protean/config.json`: an allowance of $25 per session (`max_usd`) that grows by $10 per 30 minutes of active work (`accrual_usd` per `accrual_seconds`; waiting for you accrues nothing) plus your extends, 100 ticks, $2 per thinker call, $4 per editor call (a unit or an explore), each call clipped to the least of its cap and the allowance's left, no call under $0.50 left; spend reaching the allowance stops the session and `extend` releases it. Cost is summed from the CLI's own receipts; a call that leaves no receipt, or is interrupted by Ctrl-C, is charged the cap it carried and the charge is on disk before the process exits; paid work then waits on an `unknown_spend` question that only `protean ctl answer <id> yes` releases.

`task/<task>` is the task's accepted work: it starts at the workspace HEAD when the first unit runs, every unit branches from its tip, and it only fast-forwards to a unit's commit once that unit passed and every earlier passed unit's retained file and command predicates still hold on it. A later unit may legitimately rename or split what one of those pins by naming it as a parent of its own predicates; a fresh reader call admits or refuses the hand-over, and once the children pass the parent is retired. Every retained `command` names the `protected` paths that control its result; when a later unit's commit touches one, one reader call reads the diff and says whether the checks got weaker, and a weakened candidate is rejected like a failed predicate. A unit may name `needs`; it runs only after they pass, and when one fails or is descoped it becomes `blocked`.

A unit passes only when its branch carries a change, the editor's own closing line is `exit_code: 0`, and every predicate holds: `file_exists` and `file_contains` read the branch's committed regular files (never the working tree, a symlink, a nested repo or `.git`), and a `command` predicate runs in a fresh grading clone of the workspace at the candidate (never the editor's clone) under `sandbox-exec` with no network and writes confined to the clone, uv's caches and the temp dirs. The editor process itself runs under `sandbox-exec` too, with the same write confinement and the network open: a Write, Edit, `git rm` or shell command that leaves those trees, directly or through a symlink, fails at the kernel, and without the sandbox binary no editor runs.

## Where things are

```text
src/protean/          loop · state · runner · operator · dry · cli (the ring) · graph/: director · worker · grade · budget · console · questions · reader · gate (the nodes)
brain/protean/        thinker.md, editor.md, reader.md, config.json, operator.example.json (tracked) · operator.json, state/, workspaces/ (gitignored)
fixtures/protean/     the dry scenarios
<workspace>           task/<task> (accepted work, fast-forward only) · unit/<task>/<unit> (each attempt's delivery)
tests/protean/        the battery and the shim
evidence/captures/    two zero-call CLI receipts of the earlier implementation that the ceiling and receipt tests read
architecture.md       what is built, where it lives, why it is shaped this way
docs/console-api.md   the console's contract: the snapshot, the stream, the operations, the events
```

## Design notes

**Only the grader passes a unit.** The thinker's reply never carries a status. The three ways the earlier implementation let a unit pass falsely — a misspelled predicate argument dropped from the verdict, a misspelled `file_contains` argument passing on any content, a re-plan overwriting a unit's status — are pinned as negative tests in `tests/protean/test_grade.py`, `test_director.py` and `test_loop.py`.

**Predicates read the branch, never the working tree.** A pre-existing file cannot satisfy a unit, a symlink or a nested repository is not a file, a check command runs sandboxed in a fresh grading clone, and the worker's self-reported exit code gates grading without ever substituting for it: a nonzero or missing `exit_code` line makes the attempt ungradeable, and a zero one still leaves every predicate to hold on a branch that carries a change.

**Cost is summed from receipts, never from tokens.** A capped call reports its dollars in the CLI receipt with zero tokens; a call that leaves no receipt is charged the cap it carried, and paid work waits on a question until the operator accepts the uncertainty. Every call is recorded before it launches, so a hard kill leaves a `pending` record that `resume` books once.

**The seat's cwd is stable per session.** Claude Code files a session by working directory, so `--resume` silently mints a new session unless the thinker seat always runs from the same directory. The session's seat directory is that directory.

**The app stands alone; the person wires in.** The code, prompts, tests and doctrine documents address "the operator" and read the profile at runtime; the private repository's rulings, with their names and dates, live in a state document this mirror withholds, and `architecture.md` states only the resulting rules.

**What was cut.** This loop replaced a much larger implementation: a six-node ring with per-node traces and firing grades, a second planning seat, a semantic store and intake, torn-tick replay and receipt reconciliation, think/escalate/delegate machinery, and seventy versioned schemas. Each was bookkeeping around the two steps that matter — plan a unit, verify a unit — and the rewrite landed at roughly a tenth of the size, with a battery that calls no model.

## The graph: the earlier generation

`graph/` is the implementation this loop replaced, exported from the tag it was retired at: a fixed ring of six nodes named for the brain regions whose jobs they borrow (cortex, thalamus, hippocampus, basal ganglia, anterior cingulate, homeostasis), each a folder holding its contract and its weakenable weights, with a tick loop, torn-tick replay, a kind library for contained subagents, a mailbox and offline learning. It is a self-contained project with its own README, Map and battery; run it from inside `graph/`. The loop at this root kept the two steps that mattered and dropped the rest, and `graph/architecture-logic.md` is where the reasoning behind each part lives.

## What is withheld and why

| Withheld | Why |
|---|---|
| `plans/`, `docs/STATE.md` | The planning layer: order documents, decision records, the state document with the operator's rulings by name and date, session handoffs. Process records, not code. |
| `evidence/live-*/` | The six live runs of the goal mode this loop had before thinking sessions. Their task and log records predate the receipt split and carry the operator's typed text and the machine's paths; no live-run receipt is shipped until one is captured after the split. The two files under `evidence/captures/` are zero-call CLI receipts the tests read, not run records. |
| `brain/protean/operator.json`, `state/`, `workspaces/` | The operator's profile and runtime state; gitignored in the source too. |

Nothing in this mirror was rewritten. The source carries no operator name and no machine path in its code, prompts, tests, fixtures or doctrine documents, so the export is a drop list plus this README, `CLAUDE.md` and the ignore file.

## Status

The loop reached `done` on six live runs under its earlier goal mode (receipts withheld, see above), the last of them the first live run of the build that split the loop into the `src/protean/graph/` nodes, in hold mode with a console client attached. Thinking sessions — the thought record, the five moves, the allowance that grows with active work — are built and proven dry against the battery and the scripted scenarios, and not yet exercised live. This mirror is re-exported by hand and lags the working repository. **Last updated: 2026-10-18.**
