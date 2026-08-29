# Project conventions

The canonical rules for a data-analysis project — the stable standards every notebook and
script follows. This file holds **rules**; the individual **records** they govern (data
decisions, methodology narrative, analysis reports) live in `doc/`.

**This file is the universal standard and is not copied into a project.** It is the same
for every project and is never edited to suit one of them. What a project genuinely does
vary — its path constants, its palette, whether `data/` is a symlink into shared storage,
which sources it pulls — is recorded in that project's own `doc/conventions.md`, which
states the deltas and nothing else. Read this file for the rule; read `doc/conventions.md`
for what this project chose.

Nothing here is aspirational — it describes machinery that can be lifted wholesale into a
new repository. Names in `code font` are either files to create or **illustrative
examples**; a running example of one analysis, `03_citation_impact`, is used throughout.

---

## 0. Starting a project, and starting a session

### A new project

**Create these first, then adjust the constants:**

| Create | Adjust |
|---|---|
| `src/utils/shared_paths.py` | the path constants below the header (§2) |
| `src/utils/shared_style.py` | nothing |
| `universal_settings.yml` | the palette and the per-notebook sections (§6) |
| the notebook bootstrap header (§2) | nothing |
| `.gitignore` secrets rules (§7) | the config filenames |

If a previous project already has working versions of the first two, copy them verbatim
and edit only the constants — they are deliberately generic.

**Then write `doc/conventions.md`** — this project's deltas from this file, and only the
deltas: the constants chosen, the palette, the storage arrangement (a symlink into shared
storage is a project choice, not a standard), the sources in play, and any rule here the
project deliberately departs from. A convention recorded nowhere gets re-litigated. The
*reason* for a delta belongs in the decision log (§11); the delta itself belongs here,
stated as a rule.

**Then, before the first analysis runs:** write `environment.yml` (§5), open
`doc/data/<source>.md` for each source (§10), open `doc/STATE.md` (§10), and open the
decision log (§11). These four are expensive to reconstruct later and cheap to start now.

**Do not** build the full `doc/` hierarchy, or every directory in §1, on day one. Start
with the single files §10 and §11 name, and split only when the project actually pushes
back.

If the project already has a structure of its own, **ask whether to re-set it or adopt
it** — don't silently reorganise someone else's repo.

### A new working session

A session — a fresh context, a new notebook, a collaborator returning after a month —
starts from the current state, not from the archive.

- **Load `doc/STATE.md`, the relevant `doc/data/<source>.md` entries, and the question.**
  Never "read the decision log": it is long and append-only, so it gets skimmed rather
  than read, which is worse than not loading it at all. `STATE.md` exists precisely so the
  current position fits on one page.
- **Name the sensitivity tiers in play before any data is touched** (§7), listing the
  `blind` sources explicitly. A tier declared only in a document the session never loaded
  protects nothing.
- **Alternatives and robustness checks get a clean session that has not seen the current
  approach.** A session that produced an approach will defend it; it cannot also be the
  one that tests it. Seed the clean session from the same three things and withhold the
  incumbent.
- **The ledger in `STATE.md` is authority.** A result that contradicts a ledger figure is
  reporting a bug until proven otherwise. Reconcile before building on it, and never
  quietly rewrite the figure to match the new number.

---

## 1. File structure

```
project/
├── README.md                  what this is, how to set up, how to run, pointers to doc/
├── environment.yml            the rebuildable environment (§5)
├── universal_settings.yml     the plotting style, one file for the whole project (§6)
├── config/
│   ├── <service>.ini          credentials — gitignored (§7)
│   └── <service>.ini.example  the same file with dummy values — committed
├── data/
│   ├── <source>/              pulls from one source; immutable, never edited in place
│   │   ├── cache/             per-request cache of an expensive/paid API (§8)
│   │   └── flat/              flattened exports derived from the cache
│   └── analysis/<analysis>/   derived tables, one per analysis; inherits the
│                              strictest sensitivity tier of its inputs (§7)
├── src/
│   ├── data_creation/         acquisition, dedup, matching, tagging
│   ├── data_analysis/         the numbered analysis notebooks (§3)
│   └── utils/                 the shared code layer (§2)
├── doc/
│   ├── conventions.md         this project's deltas from the standard (§0)
│   ├── STATE.md               current position: active decisions, ledger, open questions (§10)
│   ├── data/<source>.md       provenance and dictionary, one file per source (§10)
│   ├── <analysis>_methodology.md   how an analysis runs today (§10)
│   ├── <analysis>_report.md        what it found (§10)
│   ├── decisions_shared.md         project-wide decision records (§11)
│   └── decisions_<analysis>.md     records local to one analysis (§11)
├── output/
│   ├── figures/<pipeline>/<NN_analysis>/
│   └── tables/
└── logs/                      machine-generated run logs (§8)
```

**One slug, five places.** An analysis named `03_citation_impact` names, identically:

| | example |
|---|---|
| its notebook | `src/data_analysis/03_citation_impact.ipynb` |
| its modules | `src/utils/data_analysis_03_citation_impact_field_counts.py` |
| its data directory | `data/analysis/citation_impact/` |
| its figure directory | `output/figures/data_analysis/03_citation_impact/` |
| its style section | `style.notebooks.03_citation_impact` in `universal_settings.yml` |

Given any one of the five, the others are guessable. This is the convention that makes a
large project navigable — protect it when things move.

**Never hardcode an absolute path.** Paths come from the registry (§2), so the project
stays portable between a laptop and a cluster.

---

## 2. The shared code layer (`src/utils/`)

Four tiers, distinguished by name:

1. **`shared_paths.py` — the path registry.** Named constants for every directory *and
   every file that more than one notebook reads*, anchored on the module's own location
   (`Path(__file__).resolve().parents[2]`), never on the working directory. Provides
   `bootstrap()` (put `src` on `sys.path`, `chdir` to root; idempotent) and
   `ensure_dirs()`.
2. **`shared_style.py` — the style loader** (§6).
3. **`shared_<domain>.py` — reusable domain logic.** Imported by several notebooks; owns
   one subject. *Examples:* `shared_corpus.py`, `shared_taxonomy.py`.
4. **`<pipeline>_<NN>_<topic>_<job>.py` — the heavy lifting for one analysis.** Owned by
   one notebook; may have a `.sh` driver beside it (§8). *Example:*
   `data_analysis_03_citation_impact_field_counts.py`.

**The bootstrap header. First cell of every notebook, unchanged:**

```python
import sys
from pathlib import Path

# Anchor every path on the repo root regardless of where this is launched from.
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "utils").is_dir())
sys.path.insert(0, str(ROOT / "src"))
from utils import shared_paths as P
P.bootstrap()
```

**Paths are constants, not strings.** A path written as a literal in a notebook drifts,
and the drift is silent: two notebooks end up writing `data/patent/` and
`data/non_academic/patents/`, or `clinic_trials` beside `clinical_trials`, and each is
correct on its own screen. Two notebooks referring to the same file must refer to the
same constant. When a constant's meaning outgrows its name, say so in a comment at the
constant rather than renaming everything downstream.

**Secrets never live here** — see §7.

---

## 3. The notebook contract

- The first cell is the bootstrap header (§2) plus `STYLE = load_style("<notebook stem>")`.
- A notebook **orchestrates and narrates**: it loads, calls, plots, and explains. Loops
  that take minutes, parsers, and anything reused belong in a module.
- Develop inline while iterating; move a function to `src/utils/` once it settles (§9).
- A notebook must run **top to bottom in a fresh kernel**. If a cell depends on a cell
  above it having been run twice, or on a variable defined in a deleted cell, it's broken
  even though it currently works on screen.
- Long jobs are not run from the notebook — the notebook reads what the job wrote (§8).

---

## 4. Naming and dates

Two kinds of file, two rules.

**Pipeline artifacts — stable names.** Anything another notebook reads through the
registry (`COUNTS_TABLE`, `TRIALS_CSV`) keeps one filename forever. A date in the name
would force a code edit on every re-run, which is why files under `data/` generally carry
none. Instead:

- Record the snapshot date **inside** the file (a column) or in a sidecar next to it —
  that's where it's actually useful, since it survives copying.
- When a run is superseded but worth keeping, move the whole directory aside
  (`<run_dir>` → `<run_dir>_archived`) rather than date-stamping the files.
- Version history is git's job.

**Everything else — ISO date stamps (`YYYYMMDD`), sortable.** Exports that leave the
project, reports, and snapshots deliberately kept side by side: `summary_20260704.csv`.
The date is when the file was **produced**, not the data's date. Logs use a full
timestamp (`YYYYMMDD_HHMMSS`). Raw pulls keep their acquisition date and are never
re-dated or overwritten.

---

## 5. Environment

Managed with miniforge3; the environment name is close to the project name (project
`citation_atlas` → env `citatlas`).

```bash
mamba env list                      # find it
mamba activate <env-name>           # activate it
mamba env create -f environment.yml # or rebuild from scratch
```

`environment.yml` must exist and stay current. An environment that is only
*discoverable* on one machine is not reproducible anywhere else.

---

## 6. Visualization style

The style lives in `universal_settings.yml` at the repo root — **never** as a `STYLE =
{...}` literal in a notebook. Per-notebook literals are how palettes and font sizes drift
apart: nothing forces two notebooks to agree, so eventually they don't.

```python
from utils.shared_style import load_style
STYLE = load_style("03_citation_impact")   # style.base + style.notebooks.<name>
```

`style.base` holds what every figure shares (palette, `figsize*`, font sizes, `dpi`,
`save`, `formats`, `savedir`). `style.notebooks.<notebook stem>` overrides **only** the
keys that notebook genuinely does differently.

Two loader conveniences, neither guessable from the YAML alone — document them wherever
this file goes:

- In `roles` and any `*_colors` map, an **integer** means *index into `colors`*, so
  semantic roles follow the palette when the palette changes; a **string** is taken
  literally (`"#BBBBBB"`).
- Two-element `figsize*` lists become tuples, as matplotlib expects.

`savedir` is relative to the repo root and applies only when `save: true`.

---

## 7. Secrets, config, and data sensitivity

Credentials never enter a committed file, a notebook, or the registry.

- The real file is gitignored by pattern (`config/*.ini`), so a new one can't be
  committed by accident.
- A companion `<name>.ini.example` **is** committed, carrying every key with dummy
  values, so the next person knows what to supply without having to read the client code.
- Environment variables are the alternative for CI or a cluster.

Before pushing a repo for the first time, confirm the real file is actually ignored
(`git check-ignore -v config/<name>.ini`) — the pattern is easy to get subtly wrong.

### Data sensitivity

An assistant runs as a client on this machine; the model it queries does not. Files are
read locally, but everything read is transmitted for inference, and re-transmitted on
every subsequent turn until it is compacted away. **The filesystem is local; the context
window is remote.** So the boundary that matters is not what sits on disk — it is what
gets read into context.

Every source in `doc/data/<source>.md` carries a `**Sensitivity:**` field with one of
three values, and the clause of the licence or data use agreement that sets it. An
undeclared source is treated as `blind` until someone declares it.

| Tier | The assistant may see |
|---|---|
| `open` | rows freely — published or aggregate data with no redistribution restriction |
| `schema-only` | `dtypes`, row counts, `describe()`, `value_counts()` on low-cardinality columns; never raw rows |
| `blind` | nothing. Code is written against the source document alone and run by a human |

**A `blind` source needs a fixture, or the tier defeats itself.** Code that has never
executed is worse than code written from rows. Ship a synthetic frame beside the real one
carrying the same schema, dtypes, and known quirks — the mixed-level codes, the vintage
seams, the sentinel values — so the assistant can develop and test against it while the
real pass is run by hand. Building that fixture is the work the tier actually costs; the
declaration is free.

**Tier propagates to derived tables, at the strictest level of the inputs.** A `blind`
source joined into an `open` panel yields a `blind` table. Where an aggregation is coarse
enough to relax that — group means over thousands of respondents, say — the relaxation is
a logged decision (§11) naming the aggregation, not a judgement made in passing.

**Tracebacks leak.** An exception frequently embeds the offending values. Under
`schema-only` and `blind`, report the exception type and the line, not the raw traceback.

**Declaration is not enforcement.** A tier stated in a document the assistant never loaded
protects nothing. Keep `blind` paths in the assistant's ignore file, disable
run-without-approval for shell commands, and name the tiers in play at the top of the
session (§0) rather than trusting that a document three links away will be read. The
common failure is not a malicious read — it is an agent deciding on its own to `cat` a
file to understand its format.

**Contract before policy.** Many data use agreements forbid transfer to a third party
outright, and no vendor retention or training policy answers a clause like that. Check
what the agreement permits before checking what the platform permits: the first can forbid
what the second allows, and only the first ends a project.

---

## 8. Expensive work: caching, long jobs, snapshots

**Cache what costs money or rate limit.** Pulls from a paid or throttled API land in a
per-request cache (`data/<source>/cache/`) that is written once and never re-pulled;
flattened tables are derived from the cache, not from the network. Re-running an analysis
must not re-bill the project.

**One script is submitter and job body.** A long or cluster job lives in a `.sh` beside
its `.py` and takes a subcommand (`filelist`, `probe`, `submit`, `local`), so the same
file enumerates the work, sizes it, submits the array, and runs as each array task.
Everything is overridable by exported variables, with working defaults.

- **Probe before you size.** Measure one shard's time and memory before submitting
  hundreds; a `probe` subcommand makes that the easy path.
- **Partials, then a dependent merge.** Shards write per-shard, per-variant files that a
  single merge step reduces. A job that dies half way is then resumable, not restarted.
- **Cheap axes ride along.** Computing a second variant alongside the first usually costs
  far less than a second full pass over the data — bundle them into one submission.
- Run output goes to `logs/<job>_run.log`, never into `doc/`.

**Snapshot coherence.** Any two things that will be compared must come from the *same*
pass over the *same* snapshot. Where the underlying source keeps accruing — citation
counts, trial registrations — a comparison arm counted last week against one counted
today is silently wrong in a direction no one can see.

---

## 9. Modularization

Functions are small and single-purpose — one job, inputs as arguments, outputs returned,
no reliance on notebook-global state.

**Swappable backends share a schema.** When the same table can be produced two ways (a
local corpus scan and a remote API facet, say), give them the same column names and the
same filenames, so a notebook switches pathway by changing one path constant and nothing
else. Where the two are *not* equivalent — a column one pathway cannot produce — say so
in the module docstring and at the constant, because the switch is otherwise invisible.

**Module headers record the failure they prevent.** Open a shared module with a short
docstring saying what broke before it existed and what rule it now enforces — the path
registry and the style loader both earn one. That is the single highest-value comment in
a project: the code shows *what*, and nothing else records *why this exists at all*.

---

## 10. Documentation

Document what is **lost and expensive to reconstruct** — provenance, decisions, traps —
not structure the code already expresses.

### The document set

| Document | Holds |
|---|---|
| `README.md` (root) | what the project is, how to set up, how to run the notebooks in order, pointers to the rest |
| `doc/conventions.md` | this project's deltas from the standard: constants, palette, storage arrangement, deliberate departures (§0) |
| `doc/STATE.md` | the current position — active decisions, the ledger, open questions (below) |
| `doc/data/<source>.md` | provenance and dictionary for one source: where it came from, when, under what licence, its `**Sensitivity:**` tier and the clause that sets it (§7), what each column means, and its traps |
| `doc/<analysis>_methodology.md` | how the analysis runs *today*: every step, its inputs, its outputs |
| `doc/<analysis>_report.md` | what was found, the decisions at each fork, and the caveats |
| `doc/decisions_shared.md`, `doc/decisions_<analysis>.md` | the running decision log (§11) |

Generated directories (`output/`, `logs/`) get no README — their contents are dated and
machine-produced, so a static description only rots. Docs are numbered (`NN.name.md`)
when their reading order matters.

### `STATE.md`

Every other document in the set grows. `STATE.md` does not — it is **overwritten in
place** and holds only what is true now. One page, three sections.

1. **Active decisions** — one line per live entry, with its id. An index, not a summary;
   the reasoning stays in the log.
2. **The ledger** — every figure the project currently treats as established, each with
   the decision that set it and the date it was last confirmed:

   | quantity | value | set by | as of |
   |---|---|---|---|
   | country-years retained | 1,836 (79.1%) | D43 | 2026-08-21 |

   Rows are replaced, never appended to and never struck through; the history is one
   lookup away through `set by`. Thirty rows is a working ledger. Two hundred means
   figures are being recorded that nothing downstream depends on.
3. **Open questions** — what is undecided right now, and what would settle it.

The division of labour is the whole point: **the log is append-only history, `STATE.md` is
current state.** Every question about how to record a decision that revises an earlier one
resolves once those two properties live in different files. This is also the one document
exempt from *revise in the open* below — it carries no correction blocks, because its
history is the log.

### Anatomy of an analysis document

A reader should be able to stop at any depth and still have something true.

1. **Header block** — purpose in one sentence, then the source notebook, the supporting
   modules, and the scope *with hard numbers and a window* (e.g. `26,109 publications`,
   `2004–2024`), each as a **relative markdown link to the actual file**.
2. **§0 At a glance** — a table of the quantities a reader would otherwise recompute:
   row counts, coverage percentages, the headline figures.
3. **The one fact that must not be missed**, stated as such and labelled. If the study
   set is four hundredths of one percent of the reference corpus, say so before the first
   share statistic, because every later number is misread without it.
4. **A section per pipeline stage**, each ending in the decisions it baked in.
5. **Tricky points** — a recurring, named callout for traps (a source that truncates long
   list fields; synthetic ids that can merge two people). Use one consistent label so
   they can be grepped and so a reader learns to look for them.

### How to write it

- **Every number carries its denominator and share** — `26,086 (99.9%)`, not `almost
  all`. A bare count can't be checked against a re-run.
- **Record the road not taken.** "A one-row-per-author table was deliberately not used
  because it discards the per-paper context" is worth more than a description of the
  table that *was* used, and it stops the same argument being re-had in six months.
- **Name the file, the function, and the column** you're describing, and link the file.
  Prose that says "the parser" ages badly; the function's actual name doesn't.
- **Separate a property of the source from a property of the pipeline**, explicitly. It
  is the distinction readers get wrong most often, and only the author can settle it.
- **Say what a number cannot support**, next to the number. Caveats parked in a closing
  section are read after the reader has already formed the conclusion.
- Write for the person re-running this in a year with none of today's context. That is
  usually you.

### Keeping documents true

Documents rot in two directions, and both are handled by convention rather than vigilance.

- **Revise in the open.** When a document is wrong, add a dated, numbered correction
  block at the top (`> **Correction (rev. 4).** …`) saying what was wrong and what is now
  true, and carry `*Last updated: YYYY-MM-DD (rev. N).*` under the header. Silent edits
  destroy the one thing a methodology document is for — a reader cannot tell whether the
  version they remember was the mistake or the fix.
- **Supersede explicitly.** When a newer document replaces part of an older one, say in
  both which is current for which topic: "where the two disagree, this file is current;
  the findings and the two-arm design recorded there still stand." Never leave two
  documents silently disagreeing.
- **Moving a file means fixing its documents.** A report header pointing at
  `src/01_authors.ipynb` after the notebook moved to `src/data_analysis/01_authors.ipynb`,
  or at a source file replaced two formats ago, is the failure that makes readers stop
  trusting the docs. After moving or renaming anything, grep `doc/` for its old path —
  cheap, and it catches every stale header at once.
- **Retire rather than accumulate.** A document describing a pipeline that no longer
  exists is either marked superseded at the top or deleted. Keeping it "for reference"
  without a marker is how the wrong methodology gets cited.

---

## 11. Decision logging (data and methodology)

Separate **rules** (this section — how decisions are made and recorded) from **records**
(the decisions themselves, in the log).

**No silent decisions.** Every data or methodological decision follows:

1. **State the assumption** explicitly.
2. **Fact-check against the data** — show the effect: rows before/after, records affected
   and %, a sample of what changed. Confirm assumptions against the data, never on faith.
3. **Record it** (see below).
4. **Escalate** substantive or uncertain calls for a human decision before proceeding —
   ask, don't assume.

**Where records live.** In `doc/`, as curated documentation — never in `logs/`, which
holds machine-generated run output.

**A decision that establishes a project rule is recorded twice, differently.** The log
holds why the call was made and what was checked; `doc/conventions.md` holds the resulting
rule, in the imperative, where someone will look for it. A rule that exists only as the
conclusion of a log entry will not be found by the person who has to follow it.

**How records are organised — partition by scope, tag by kind:**

- _Scope sets the file:_
    - `doc/decisions_shared.md` — decisions that apply project-wide (ingestion, global
      standardisation, canonical dataset construction). Made once, reused.
    - `doc/decisions_<analysis>.md` — decisions local to one analysis. The same dataset
      may be handled differently across analyses; each analysis's choices live in its own
      file.
- _Kind is a tag on each entry_ (`data` or `method`), not a separate file — so data or
  method decisions can be filtered across files without the kind axis fighting the scope
  axis.

**Entry shape.** A header line carrying **date**, **kind**, and **status**, then the
assumption, the check that was run and what it showed (with numbers), the decision, its
**effect**, and the alternative rejected. An entry without the check is a note rather than
a record; an entry without the effect leaves the ledger nothing to read.

```
## D43 — A 0.07 discrepancy threshold between coherent and direct
**Date:** 2026-08-21 · **Kind:** `method` · **Status:** `active` · **Scope:** `05`, `06`
```

**Status is what makes the log safe to read out of order.** Four values: `active`,
`superseded by <id>`, `partly superseded (<what> → <id>)`, `retired`. Stamping the
superseded entry is the mandatory second half of writing a new one — an entry that
supersedes without stamping leaves a live-looking rule that someone will act on, and the
most-referenced entries are the likeliest to have been partly overtaken. The body of a
superseded entry is never edited; only its status line changes. Cross-references are by
id, so ids are permanent and entries are never renumbered.

**Kind** is `data`, `method`, or `bug` — and a bug is logged only when it changed a figure
someone had already seen. Everything else git records better.

**Write the entry when the decision is made, not at the end of the session.** It goes into
the notebook as a markdown cell, in log format, at the moment the call is made.
Reconstructing a session's decisions afterwards reliably preserves the conclusions and
loses the conditions attached to them, which is the half that matters. Moving those cells
into the log is the end-of-session job; remembering is not.

**Start simple, scale up.** A small project needs only a single `doc/decisions.md`. Split
it into `doc/decisions_shared.md` plus `doc/decisions_<analysis>.md` only once analyses
genuinely diverge — don't build the hierarchy prematurely. `STATE.md`, by contrast, is
worth opening on day one however small the project is: it is one page, and it is what a
new session reads.

**Methodology narrative.** `doc/<analysis>_methodology.md` is the readable synthesis of
the `method`-kind decisions plus assumptions and limitations, for write-up. Method
decisions are still logged like any other; the narrative reads from them, and the log
stays the record of record.

---

## 12. Reproducibility

Any step involving randomness — sampling, splits, network layout, shuffles — sets an
explicit seed, so results are reproducible run to run. The seed is a **named parameter of
the function that uses it**, with a default at the top of its module
(`def sample_balanced(..., seed: int = 42)`), never a bare literal buried in a call —
`nx.spring_layout(G, seed=42)` inside a plotting function is a seed no one can find or
change. Where an analysis has several, collect them as module-level constants
(`RANDOM_STATE`) and read them from there.

The same applies to anything time-dependent: pin the snapshot (§8) and record which one
was used.
