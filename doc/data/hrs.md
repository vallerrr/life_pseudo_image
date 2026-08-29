# Source: HRS (Health and Retirement Study)

*Last updated: 2026-08-29 (rev. 1).*

**Sensitivity:** `blind` — **provisional.** No clause has been located and recorded yet,
so the standard's default applies ([conventions §7](../00_project_conventions.md):
"an undeclared source is treated as `blind` until someone declares it"). This is the
conservative reading, not a confirmed one; see *Settling the tier* below. Tracked as
D07 and as open question 1 in [`STATE.md`](../STATE.md).

**Status in this project:** not yet used. Nothing in the current pipeline reads it. The
only code that ever touched it is the retained prototype
[`src/legacy/hrs_preprocess.py`](../../src/legacy/hrs_preprocess.py), which is not part
of any pipeline and is not run.

---

## Provenance

- **What it is.** The Health and Retirement Study, a longitudinal panel of US adults over
  50, plus the RAND HRS Longitudinal File, which harmonises the raw waves into one
  cleaned respondent-level file.
- **Where it came from.** Downloaded from the HRS distribution site by the project owner.
  The exact download date is **not recorded** and cannot be reconstructed from the files;
  directory mtimes read 2026-07-24, which is when the repository was reorganised, not
  when the data was acquired. Record the acquisition date the next time a pull is made.
- **Where it lives.** `data/hrs/`, registered as `P.HRS_DIR`. 4.8 GB, gitignored
  wholesale — the pattern is `/data/hrs/` in [`.gitignore`](../../.gitignore).
  Immutable: never edited in place, never re-dated.

## Contents

Eighteen directories, each a Stata export. Names encode wave and release, so the release
letter is part of the provenance and must not be normalised away.

| directory | wave / content | size |
|---|---|---|
| `randhrs1992_2022v1_STATA` | RAND HRS Longitudinal File, 1992–2022, v1 — the harmonised file | 1.6 GB |
| `h92f1b_STATA` … `h22e3a_STATA` | raw core waves, 1992 through 2022 | 74 MB – 337 MB each |
| `a93f2a_STATA`, `a95f2b_STATA` | AHEAD 1993 and 1995 cohorts | 38 MB, 74 MB |
| `hd10f6b_STATA` | 2010 exit/other supplement | 282 MB |

Each directory ships a `.dta` and its codebook PDF (`randhrs1992_2022v1.pdf`,
`README_fat22.pdf`, …). **The codebooks are the intended reference under a `blind`
tier** — code is written against them, not against rows.

## Dictionary

No dictionary is recorded here yet, because no variable has been used yet. The one
existing selection is the prototype's, in
[`src/legacy/legacy_params.py`](../../src/legacy/legacy_params.py), which picks RAND
wave-1 variables into three SDOH streams (`upper` / `middle` / `lower`) and carries a
`replace_dict` of hand-built recodes. Treat that as a starting point that has never been
validated, not as a specification: it is wave-1 only, its recodes reverse the sign of
several binaries, and nothing checks it against the codebook.

Write the real dictionary here — one row per variable, with its RAND name, its meaning,
its coding, and its traps — when the ingestion work starts.

## Traps

- **Release letters are not interchangeable.** `h12f3a` and `h14f2b` are different
  releases of different waves; a glob that sorts them lexically will not sort them
  chronologically (`h00…` precedes `h92…`).
- **RAND is derived, the core files are raw.** A variable present in both may not carry
  the same coding, and RAND imputes where the core file leaves a gap. Say which one a
  number came from; this is a property of the *source*, not of any pipeline built on it.
- **Wave-varying availability.** A variable existing in the RAND file does not mean it
  exists in every wave; the prototype's `.dropna()` silently deletes respondents rather
  than reporting the gap.

## Settling the tier

To move off `blind`, find and quote here the clause of the HRS / RAND data use agreement
that governs transfer of data to a third party, then set the tier accordingly. Two things
follow, and they are the reason this matters more than the paperwork suggests:

1. **Contract before policy.** If the agreement forbids transfer outright, no vendor
   retention or training policy answers it. Check what the agreement permits before
   checking what any platform permits.
2. **A `blind` source needs a fixture or the tier defeats itself.** No synthetic frame
   matching the RAND schema exists yet. Until one does, code written against this source
   will never have executed before a human runs it — which is worse than code written
   from rows. Building that fixture is the real cost of the tier, and it is currently
   unpaid.
