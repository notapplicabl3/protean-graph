# PROTEAN — architecture

What is built, where it lives, and why it is shaped this way. The front door is `README.md`; the console's contract is `docs/console-api.md`; the state document, the plan and the design records are kept in the private repository, not here. This file is the layer between the front door and the code: the what/where/why of the running system.

## The idea

A thought goes in. A model seat, the **thinker** (the director seat renamed in prose; its key stays `director`), answers each call with exactly one move plus a `state` (where the thought stands in at most five plain sentences, and what is open): `ideas` lays out 3–7 options and waits for the operator; `explore` sends a sandboxed **worker** to try a possibility in a clone, whose branch `explore/<session>/<n>` comes back with an unverified report and is never graded or integrated; `task` plans small units, each with checks a program can run; `ask` asks the operator; `rest` waits for their next thought. For each unit the worker edits a clone on its own branch and commits. A **grader** runs the checks against the committed branch; only the grader can mark a unit passed. Accepted work accumulates on one branch in the workspace, `task/<session>`, which only moves to a verified commit. The brain alone writes the thought record (`thought`, `standing`, `open`, `decided` — from the operator's `think --decide` only — and `moves`) inside `task.json`, rendered to `thought.md`. A session never ends: it parks on a question or a wait for a thought, and stops only at a dollar or tick ceiling. `main` is never touched; nothing is merged or pushed. That is the whole program (thinking sessions, per their spec in the planning layer; before them a goal ran to `done`).

Three trust levels make it work: the thinker is trusted to think and plan and never touches a file; the worker touches files and is trusted with nothing; the grader judges both and calls no model.

## One session, end to end

1. `protean think "<thought>" [--workspace <clone>]` creates `brain/protean/state/<session>/` (and, with no workspace, a one-commit scratch repo under `brain/protean/workspaces/<session>/`), takes a lock, and starts ticking. A parked session does not block a new one; a live brain refuses a second.
2. Each tick runs the steps of `loop.TICK`, one declared sequence the loop executes: repair (finish any pass cut between its two writes; end the tick if the task is terminal, a question is open or something awaits `approve`); block units whose needs died; check the ceilings; release a held plan once its branches are admitted (after the ceilings, so an approved plan's reader call is never paid at one; a think that landed meanwhile returns it instead); pick the first pending unit whose needs are all passed and integrated, or a pending explore; hold a retry or an explore for `approve` in `hold` mode; seat (nothing pending: call the thinker, else the worker on that unit or explore); block again; count the tick and save. A session awaiting a thought (after `ideas` or `rest`) calls nothing until a `think` lands.
3. The thinker's reply is exactly one move plus `state`: `ideas`, `explore`, `task` (units), `ask` or `rest`. A `standing` byte-identical to the last accepted move's is no progress: returned once, then a runtime `no_progress` question holds every paid call until `yes`. A unit whose predicates name `parents` proposes a branch: the gate hands the parents, the children and the director's rationale to a fresh **reader** call, directly, never through the director; a refusal returns to the director as an error, an admission lets the unit run, and at its integration the parents leave the re-check set for good. A malformed reply is retried once with the errors, then parks the task. A reply whose receipt confirms the call hit its dollar cap (`subtype: error_max_budget_usd`) is a **ceiling**, not a malformed plan: no retry on that input, a `ceiling` log event names the effective cap and which constraint bound it (the per-call limit or the task's remaining budget), and the task parks on a `runtime:` question that only `resume --extend` releases — an extension widens the task budget, never the per-call cap.
4. The worker runs in a fresh clone with no remote, on `unit/<task>/<unit>` branched from the accepted tip, under a kernel sandbox. It ends with its own `exit_code: N` line. The branch is fetched back to the workspace (`--no-tags`), and the candidate, what changed and whether it extends the tip are read there, never in the editor's clone.
5. Grading happens only when the worker reported `exit_code: 0` and the branch carries a change. Every predicate of the unit is checked in the grading clone, a fresh clone of the workspace at the candidate that is removed afterwards (never the editor's own clone, whose `.git` and ignored files the editor wrote), then every accepted unit's retained predicates are re-checked in that same grading clone; a parent whose children all passed on this candidate is excused from that re-check, not graded. When those free checks pass and the candidate touched a `protected` path of an accepted retained `command`, one reader call reads the hunks and says whether the checks got weaker; weakened, or no verdict, rejects the attempt like a failed predicate. All pass → the unit is `passed` with its candidate commit, then `task/<task>` moves to it and the unit is `integrated`. Any failure → the attempt is rejected with notes; two rejections → `failed`, and dependents become `blocked`.
6. A question (the thinker's `ask`, or the runtime's — unknown spend, a cap hit, a reply invalid twice, no progress) is recorded in `task.json` (`questions`: id, asker, reason, status, answer, actor) and emitted on the console; the brain WAITS at the tick boundary for `answer` (routed to the asker only — a runtime answer never reaches a director message), `extend` or `stop`. With no client attached for `unattended_seconds` it fires one macOS notification and parks `interrupted`, exit 12; `protean ctl answer` works offline against `task.json` and `protean resume` continues on the same director session.
7. An explore runs through the same worker seat in a fresh clone at the accepted tip, under the same sandbox and cap; its branch is fetched back, its report (the tail of the worker's output) is stored on the move labelled unverified, and nothing grades, retries or integrates it. There is no `done`: `rest` records the move and the session waits for a thought.

Exit codes: stopped 11, interrupted (parked on a question or a thought) 12, interrupted mid-call 130.

## The tree

```
src/protean/
  cli.py        think · resume · status · dry · ctl; exit codes
  loop.py       the tick as data (`TICK`: repair → block → ceiling → release → pick → retry → seat → block → save); the two-step pass; re-checks; integrate
  state.py      Task, Unit, Predicate; task.json + log.jsonl per task; the scheduler (pending_unit, block_dependents)
  runner.py     the one subprocess seam for model calls; call cap clipping; the $0.50 floor; process-group kill
  operator.py   the operator: one profile file read at runtime (`operator.json`, else the tracked example, else a default); a seat is told name and preferences, never the workspace path
  dry.py        scripted seats so the battery and `protean dry` never reach a model; `--serve` is the fixture brain the console app attaches to (starts paused, waits on holds, paced, transcript lines)
  graph/        the nodes, one module each; the ring above stays at the root
    director.py the planning seat: one JSON message in, one JSON reply out; decode + validation; the resumed session
    worker.py   the editor: clone, branch, tool grant, sandbox profile, the exit line, fetch-back; WorkerResult
    grade.py    predicates (file_exists, file_contains, command; exit_code legacy); branch-tree reads; the command sandbox; writable_trees
    budget.py   Caps: the allowance (`max_usd` 25 + `accrual_usd` 10 per `accrual_seconds` 1800 of active work + extensions), ticks, per-call caps, walls, attempts, the model pinned per seat; ceiling(); accrued_usd / allowance / call_caps; cost + tokens from receipts; the per-call record (open_call / resolve / mark_booked) and resume recovery
    console.py  the console: 127.0.0.1 server thread (per-run token, 0600 console.json), snapshot + NDJSON stream, operations pause/resume/stop/think/answer/extend/approve/mode landing in state, the tick-boundary waits, the unattended park, osascript notify
    questions.py the floating runtime-questions node: deterministic handling of runtime-asked questions (unknown spend, cap hit, reply invalid twice, no progress, stale tip); no model, no import of director
    reader.py   the reader seat: one fresh read-only call per branch proposal or protected-path touch; a brief in, yes/no + kind + note out
    gate.py     branch admission (validates parents, calls the reader, excuses parents on the branching candidate, retires them at integration) and the weakening check (protected set, touched paths, bounded hunks, one reader call); no model call
brain/protean/
  thinker.md    the thinker's prompt             editor.md   the worker's prompt            reader.md   the reader's prompt
  config.json   caps + the model per seat        operator.example.json   the profile a fresh clone runs on (`operator.json` beside it is gitignored)
  state/      per-task task.json, log.jsonl, transcript.jsonl, console.json, seat session, lock  (gitignored)
  mailbox/      closed items of runs 1–5 (legacy)  workspaces/ throwaway clones for live runs                   (gitignored)
fixtures/protean/   dry scenarios (think, think-rest, dry, stuck, deps, deps-broken, deps-blocked, deps-delivery-only, branch, weaken); the fake `claude` shim is `tests/protean/shim/`, the predicate shim `src/protean/shim/`
tests/protean/      the battery; a shim on PATH proves no model call
evidence/           CLI receipts the tests read (`captures/`); the live-run receipts are kept privately
docs/console-api.md   the console's contract: the snapshot, the stream, the operations, the events
```

Model calls exist in exactly three places, `graph/director.py`, `graph/worker.py` and `graph/reader.py`, through `runner.py`. Everything else is deterministic.

## The parts and what each may do

| Part | Job | Reads | Writes | May never |
|---|---|---|---|---|
| thinker (key `director`) | one move per call — ideas, explore, task (units with predicates), ask, rest — plus `state` (standing, open) | the thought record, every thought in order, the workspace path (Read/Grep/Glob), all units + statuses + worker summaries, the moves, budget left, the answer to its ask | one JSON reply | edit a file, mark passed, merge/push, answer its own question, write `decided` or `moves`, repeat a `standing` |
| worker | one unit in a clone, commit | intent + predicates; on attempt ≥ 2 the previous attempt's grader findings (failed/ungradeable ids with notes, `broke` lines, the commit evaluated) as a delimited evidence block, never the operator's thoughts; for an explore, the thinker's brief | commits on its unit or explore branch; a summary or report labelled unverified | push, fetch, add a remote, write outside its trees, be graded on its word |
| grader | verdict per predicate | the unit, the clone at its candidate, retained predicates of accepted units (minus the parents the candidate's own passing children excuse) | the verdict | pass on a claim, pass with an ungradeable predicate, grade a dirty clone, use the network |
| reader | admit or refuse one branch of retained predicates; say whether a candidate weakened protected checks | a branch: the parents with the commit each last passed on, the children, the rationale, a read-only clone at the accepted tip. A weakening check: the unit's intent, the accepted commands at stake, the touched paths and bounded hunks, a read-only clone at the candidate | yes/no + a note (+ the kind of change), as a `branch_admitted`/`branch_refused` or `weakening` event | edit, plan, talk to the director, be asked twice about the same branch, be asked about a candidate that touched nothing protected |
| budget | the allowance, ceilings and caps; the per-call record | session cost, tick and `active_seconds`, caps, extensions, receipts | the stop reason; per-call caps clipped to the least of the per-call cap and the allowance's left; one `Call` per model call in `task.json` + a `call` log event (charge vs receipt cost, tokens, message size, session, outcome; unavailable = null) | widen itself, estimate from tokens, record an unavailable measurement as 0 |
| console | the human channel, live | operations from the terminal client (`protean ctl`), with the token | questions, thoughts, `decided` (from `think --decide` only), paused, mode, approvals, extensions in `task.json`; a log event per operation (`actor: console`); the snapshot + stream out | bind beyond 127.0.0.1, accept an operation without the token, treat a timeout as an answer, let an operation reach a node directly, widen a cap except by `extend` |
| runtime (loop + state + runner) | the ring | persisted state, process results, receipts, git state | task state, log, clones, `task/<id>`, the lock | call a model, decide a verdict, widen a cap, merge or push |

## The invariants that make a pass trustworthy

The old tree's `passed` could not be trusted (three reproduced false-success paths). Every rule below exists because a run or a review showed its absence let a false pass through.

- An ungradeable predicate counts against the unit, never for it. A unit with no predicates never passes. Only the grader sets `passed`.
- File predicates read the branch's committed tree objects, not the working directory, with a size bound and realpath confinement. Symlinks, ignored paths, nested repositories and files a check command wrote are missing or ungradeable.
- An unchanged branch is undelivered: a predicate satisfied by pre-existing files cannot pass a unit.
- The worker's own closing line gates grading: nonzero or missing means nothing is graded. Zero admits the candidate to verification and proves nothing.
- Grading, re-checks and the weakening diff run in the grading clone, a fresh clone of the workspace at the candidate; the brain never trusts the editor's `.git` (a replace ref, a graft, an fsmonitor or clean filter in its config, an ignored `.venv` file), every brain-side git command runs with the fsmonitor off, replace objects ignored and no system config, and `git status` in an editor-written repository runs inside the check-command sandbox. After the check commands the grading clone must be clean and still on the candidate, or the grade is void.
- A candidate must extend the accepted tip; it is rejected if it drops it or breaks any accepted unit's retained predicate (`broke <unit> <pid>`).
- A pass is two durable writes; a tick or resume repairs a pass cut between them without a model call. `task/<id>` moves by a checked `update-ref`; a ref moved from outside parks the task on a question.
- Needs are validated at decode: no self-need, no unknown unit, no cycle, no need on a unit that predates the accepted branch. File-predicate paths are validated at decode with the grader's own confinement rule (`grade.confined`: relative, inside the tree after realpath, not `.git`, no `..` escape) — an invalid path is a correctable decode error returned to the director, never a paid worker call; the grader applies the same rule at grading, so an absolute path or a symlink escape is ungradeable there too.

## Containment and money

- The **worker** runs under `sandbox-exec`: writes denied everywhere except the clone, uv's caches, the temp dirs and `/dev`; fail-closed if the binary is missing; network open (a documented limitation: one profile covers the model call and its children). Tool grant: Read, Edit, Write, Grep, Glob, `uv`, and `git status/diff/add/commit/checkout/branch/rm` — `Bash(uv:*)` admits `uv run sh -c …`, so the grant is a general shell in practice and containment rests on the kernel sandbox alone, never on the grant. Permission mode "don't ask", so anything outside the grant simply fails. `~/.claude` is outside its trees; the CLI tolerates that.
- The **check command** runs under the same profile plus network denied; both profiles deny reading the task's `console.json` (the console token), proven on the real binary.
- Every model call: `claude -p --setting-sources "" --model <the seat's pinned model, `config.json` `director_model` / `worker_model` / `reader_model`, default claude-opus-5>`; the director with `--output-format json`, the editor with `--output-format stream-json --verbose` so its transcript streams to the console line by line (`state/<task>/transcript.jsonl`, never `task.json` or a director message) and its receipt is the final `result` event. The director's tools are restricted with `--tools`; `--allowedTools` only pre-approves.
- **Ceilings:** a session's **allowance** starts at $25 (`max_usd`) and grows by $10 per 30 minutes of **active work** (`accrual_usd` / `accrual_seconds`; active = wall time the brain spends driving a tick — calls, grading, re-checks — never time waiting at a tick boundary or parked; each tick's time is added to `active_seconds` at its save, a tick cut before its save forfeits it), plus the operator's `extend`s; 100 ticks; $2 per thinker call, $4 per worker call (units and explores), 5 and 15 minute walls, two attempts per unit. Spend reaching the allowance stops the session (`max_usd:` stop reason) in the save that books the crossing call, and `extend` releases it; each call's cap is clipped to the least of its per-call cap and the allowance's left, and under $0.50 of that, no call starts. Cost is summed from receipts; a call with no receipt is charged its cap; an interrupted call books its spend before the signal propagates; a hard-killed call is recorded `pending` before launch and `resume` books it once at its reserved cap and marks the actual spend `UNKNOWN`. (This rule replaced a per-sitting cap.)

## Task size

There are no size classes. Nothing classifies a thought or a task move as small, medium or complex, and no cap, prompt or code path changes with size; a one-line fix and a multi-module feature enter the same loop under the same ceilings. Size exists only as consequences.

- **Where size is decided.** Once, by the thinker, at the unit level inside a `task` move: the prompt asks for units "small enough for one worker call (a few files, one check)" and gives no other rule. The thinker may plan incrementally — a first `task` move of a few units, more moves after they land — or lay the whole plan in one move; nothing in the runtime prefers either.
- **How a wrong size surfaces.** By failure, after the money is spent. A unit too large for one worker call fails its attempt, fails again, comes back `failed`, and the thinker is told to give its work again smaller in a new task, explore what stopped it, or ask the operator. An explore that runs long simply hits the worker's cap and lands as `lost`/`failed` on its move. A thought too large for one session surfaces as a ceiling stop.
- **What varies with size.** Units and the depth of their `needs`; moves; ticks — each thinker call and each worker call (unit or explore) is one tick, so a session's tick count is its moves plus its worker calls plus retries and questions; dollars — the thinker's per-call cost grows with its resumed session ($0.07 → $0.93 over five calls in one measured goal-mode run); the allowance ($25 + $10 per 30 min of active work) binds when spend outruns active time, and 100 ticks bind a long session.
- **What the ceilings mean in practice.** $25 plus accrual and 100 ticks per session: at measured prices (a thinker call about $2, a worker call up to $4) the base buys roughly eight to twelve moves, and half an hour of active work adds about three more; a session that mostly waits for the operator accrues nothing. Measured cost per move from a live thinking session is owed before these numbers are trusted.
- **Where a large thought breaks.** The fixed ceiling (one shape for every size); the thinker's per-call cost growth under the resumed session with no reset in a session that never ends; serial workers; the absence of any carry between sessions beyond `task/<session>` and the thought record. None of these is a size mechanism failing — there is none; they are the costs of one shape.

A sizing step, if one is ever wanted, has a natural home at the gate (admission: is this thought one session?) and would be a design item in the graph design record, not a fact of this file.

## The graph

PROTEAN is a graph in three senses, named in the graph design pass.

- **The system graph:** the parts above are nodes, the tick order is the edges, declared once as the data `loop.TICK` that the loop executes. Present: director (cortex), worker, grader, budget, console (brain side; the mailbox it replaced is gone), runtime, and the gate (a name for the admission checks spread across decode, scheduler and budget). Absent by contract, with triggers: thalamus (what the director is shown) and hippocampus (memory across tasks on one workspace).
- **The work graph:** units and their `needs` inside a task. The scheduler, not list order, decides what runs; a dead need blocks its dependents transitively. Built and proven live.
- **The memory graph:** nothing exists. The goal-mode task logs, their direction lists and closed questions persist; nothing reads them.

The contracts, the rejected alternatives, and the ordered list of builds they imply are in the graph design record. None of those builds is authorized.

## Why this shape

- **Rewrite, not carving.** The old six-node ring (the retired tree, kept at its tag in the history and exported as `graph/` in the public mirror) was ~23k lines whose completion signal was untrustworthy. It was ruled "too much code and doctrine for a relatively simple idea". The loop kept the two steps that mattered, plan and verify, and dropped the ring, the semantic store, torn-tick replay, learning and the oracle.
- **Expansion only from observed need.** Every addition since (dependencies, the accepted branch, the exit gate, `git rm`, the sandbox) came from a live run showing a gap, and landed with its repro test first, one fresh review after.
- **Serial by design.** One worker per tick. Parallel work waits on budget reservation.
- **Resumed director session.** The director remembers for free and its cost climbs every call ($0.07 → $0.93 across one task). Kept pending token measurement; a thalamus is built only on demonstrated pressure.
- **The battery is the oracle.** `uv run pytest` runs in about 70 s (bound 90 s by pytest's reported time) with a shim proving no model call; `protean dry <scenario>` replays ten scripted runs, and `--serve` serves one for a console client. Every change is checked against it.

## Proven live, proven dry, unproven

- **Live** (receipts of the six goal-mode runs, kept privately): `done` on a real goal; a question, an answer, a resume in a new process on the same session; three units with needs running in dependency order with the accepted branch advancing; a failed attempt, a unit rejected for breaking a retained predicate, a `blocked` dependent, a descope on a human yes, the director refusing `done` on an unmet goal; a sandboxed editor doing `git rm`, tests and a commit, and its one outside write refused at the kernel.
- **Dry only:** a worker hitting its dollar cap or wall timeout; the director being refused a `Bash` call.
- **Unmeasured:** final-turn overshoot of a call cap (no call has hit one); the split of director cost between message size and transcript.

## Known limits

All check commands of one candidate share one grading clone, so the unit's own check could plant something (a `.venv` `.pth`, a forged loose object, a skip-worktree bit) that a later re-check in the same pass reads; closing it would mean a clone per check or re-checks before the unit's own commands. A recorded known limit, kept as built until a run shows it matters. No sizing: one loop, one ceiling, every thought; a thought larger than one session is split by hand with nothing carried between sessions (§ Task size). Open worker networking. A retained predicate pins what it checks for the rest of the task unless the director branches it to children a reader admits; `retain: false` marks a delivery-only check. A weakening check rests on one reader call's judgment of the hunks and on the director naming the right `protected` paths; a path it forgot is unguarded. A hard-killed run's editor outlives the parent with no wall and may still write into the clone the next attempt rebuilds.
