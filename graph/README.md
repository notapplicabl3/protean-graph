# PROTEAN graph

The earlier generation of PROTEAN: a fixed graph of six nodes around Claude Code, with memory, budget checks, gates, feedback and offline learning. It was retired on 2026-09-27 in favour of the small loop at this repository's root, and it is kept here because the graph is the design the loop was distilled from.

This directory is exported from the tag the tree was retired at. It is a self-contained project: run it from here, not from the repository root.

## Why it exists

A language model can plan work, but a long conversation does not itself provide durable task state, a budget signal or a reliable comparison between expectations and results. This generation supplied those surrounding mechanisms as one graph shared by every project.

Two model seats direct the work: a director sets direction and a manager conducts specialized subagents. The other nodes are deterministic: the thalamus bounds what a seat is sent, the hippocampus retrieves evidence, the basal ganglia gates what runs, the anterior cingulate grades expectation against outcome and watches for over-firing, and homeostasis holds the ceilings. Offline learning (`sleep`) turns repeated evidence into adjusted weights, reusable procedures and project memory.

## The graph

```text
brain/nodes/<node>/     NODE.md (the contract) · weights.yaml (the seeded, weakenable priors) · procedures/
src/protean/nodes/      the deterministic node bodies, a registry, detectors and the shared vocabulary
src/protean/cortex/     the two seats, the kind library, live invocation and containment
src/protean/runtime/    the tick loop: checkpoint plus journal per tick, torn-tick replay, terminals, archive
src/protean/state/      the persisted contracts and their schemas
src/protean/sleep/      offline learning and its report
src/protean/intake/     seeding stores from an external policy home, offline, between tasks
src/protean/oracle/     the synthetic-dev oracle: predicates graded against a cloned workload
src/protean/mailbox/    questions to the operator; silence is never assent
brain/seats/            the director and manager prompts, the call prompts, the kind library
fixtures/               scripted scenarios, envelopes, traces and workspace seeds
tests/                  the dry battery and a recording stand-in for the model
docs/                   doctrine and operations
```

`architecture.md` is the full tree with ownership; `architecture-logic.md` records the design choices and the alternatives rejected.

## Run it

Python 3.13 or newer and `uv`. No model account is needed for the dry path.

```sh
cd graph
uv sync --locked
uv run pytest
uv run protean dry dry
uv run protean dry stuck
```

The dry scenario runs a scripted task from goal to completion in temporary copies and calls no model; a stand-in `claude` on PATH proves it. `stuck` ends at an interrupt rather than success. The six verbs (`run`, `resume`, `status`, `dry`, `intake`, `sleep`) and their effects are in [docs/operations.md](docs/operations.md).

**The battery from a fresh clone.** The layout tests assert that the generated brain directories and each node's trace file exist, and the source keeps those untracked, so create them once before `uv run pytest`:

```sh
mkdir -p brain/state brain/episodes brain/projects brain/mailbox/open brain/mailbox/orphaned
for n in brain/nodes/*/; do touch "$n/trace.jsonl"; done
```

Four tests then still fail, and are expected to: three read a real archived run (`tests/sleep/test_habits.py`), and one runs `git show` on an index path that assumes this directory is the repository root (`tests/state/test_brain_tree.py`). Everything else passes.

## What is withheld and why

| Withheld | Why |
|---|---|
| `plans/`, `reports/`, the status document | Specifications, work orders, audits, session records and gate ledgers: process, not code. Comments cite them by bracketed tag; the tags stay, the documents do not. |
| the live drivers and their captures | Every live run used one private repository as its workload, and its receipts carry that repository's paths, tree digests and session transcripts. Two probe drivers (`tests/wet/probe_library.py`, `probe_triggers.py`) and three zero-call receipts under `tests/cortex/captures/` stay because tests read them; their paths were rewritten for publication, so the sandbox profiles they show would not run as printed. |
| the dashboard prototype | Its static page embedded a row of a private registry. Nothing else imports it. |
| one recorded session fixture | A CLI transcript over the private workload. |

The renaming was consistent: the operator's name (including the `operator_answer` signal), the workload's name (now `workload`; the oracle source keeps the workload's own command names, `rig` and `digest`), the intake module named for the operator's configuration repository (now `policy_home`, with that repository's file names replaced by generic ones), and a data directory the oracle reads in the workload clone (now `.notes`). No behaviour changed beyond those names. Comments also cite a design digest by number (`DIGEST:20`, `digest §2.8`); the digest is not in this mirror and the citations stay for the same reason the bracketed tags do.

## Status

Retired. Built and dry-proven; the live gates it was waiting on were never opened, and its completion signal was shown untrustworthy, which is what prompted the rewrite at the repository root. **Last updated: 2026-10-18.**
