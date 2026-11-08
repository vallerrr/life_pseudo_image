# Source: HRS (Health and Retirement Study)

*Last updated: 2026-10-25 (rev. 5).*

**Sensitivity:** `blind` — **provisional.** No clause has been located and recorded yet,
so the standard's default applies ([conventions §7](../00_project_conventions.md):
"an undeclared source is treated as `blind` until someone declares it"). This is the
conservative reading, not a confirmed one; see *Settling the tier* below. Tracked as
D07 and as open question 1 in [`STATE.md`](../STATE.md).

**Status in this project:** ingestion written, **never run on the real file.**
[`src/utils/shared_hrs_ingest.py`](../../src/utils/shared_hrs_ingest.py) turns the RAND
file into model-input arrays using the dictionary below; it has run only on the synthetic
fixture [`tests/fixtures/rand_hrs_fixture.py`](../../tests/fixtures/rand_hrs_fixture.py),
tested by [`tests/test_hrs_ingest.py`](../../tests/test_hrs_ingest.py). The first real
pass is a human's to run: `write_model_input(ingest_rand(), hrs_model_input_path())`,
which writes to `P.HRS_MODEL_INPUT_DIR` (as `blind` as its source). The choices below are
D15–D21 in [`decisions_shared.md`](../decisions_shared.md). The retained prototype
[`src/legacy/hrs_preprocess.py`](../../src/legacy/hrs_preprocess.py) is superseded by the
ingestion module and is not run.

**How results come back** (D22): the user runs a script, and the script writes one
aggregate JSON report under `output/reports/blind/`, which is the only thing an assistant
reads. There are two such scripts:
[`01_hrs_schema_report.py`](../../src/data_creation/01_hrs_schema_report.py) reports the
file's structure, and [`04_hrs_chronic_cnn.py`](../../src/data_analysis/04_hrs_chronic_cnn.py)
`counts` / `run` report the chronic-disease task (D23). The protocol is in
[`conventions.md`](../conventions.md), "Blind sources: run and report".

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

Every variable the ingestion reads, from the RAND HRS Longitudinal File 2022 (V1)
codebook (`randhrs1992_2022v1.pdf`; page numbers in brackets). Nothing here was read from
the data. The code-side copy is `CHANNELS`, `STATIC_COLUMNS` and `WAVE_STEMS` in
[`shared_hrs_ingest.py`](../../src/utils/shared_hrs_ingest.py); a test fails if any name
it reads is missing from this page.

**Names.** The codebook writes names in upper case with `w` for the wave (1–16):
`RwBMI` is `R1BMI` … `R16BMI`. `R` is the respondent, `H` the household, `RA`/`HA` not
wave-specific [14]. The prototype lower-cased names before selecting from the Stata
file, which suggests the file stores them in lower case; the reader matches either.

**Storage types are not documented.** The codebook gives only `Cont` / `Categ` / `Char`
[47], not Stata byte/int/float/double. The reader does not depend on them.

The prototype's selection in [`legacy_params.py`](../../src/legacy/legacy_params.py) is
superseded: it was wave-1 only, its recodes reversed the sign of several binaries, and it
was never checked against the codebook.

### Waves and calendar

| wave | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| year | 1992 | 1994 | 1996 | 1998 | 2000 | 2002 | 2004 | 2006 | 2008 | 2010 | 2012 | 2014 | 2016 | 2018 | 2020 | 2022 |

For the AHEAD cohort, waves 2 and 3 are 1993 and 1995 ("2A", "3A"), on a different
instrument from the HRS cohort's 1994 and 1996 ("2H", "3H") [13]. A wave number is not a
calendar year.

Entry cohorts (`HACOHORT`) [12–13]: HRS (3, born 1931–41) in wave 1; AHEAD (1, before
1924) in wave 2; CODA (2, 1924–30) and War Babies (4, 1942–47) in wave 4; Early Baby
Boomers (5, 1948–53) in 7; Mid (6, 1954–59) in 10; Late (7, 1960–65) in 13; Early Gen X
(8, 1966–71) in 16; HRS/AHEAD overlap cases (0) from wave 1. Spouses join regardless of
birth year, so some are under 50 at entry. Fieldwork can run into the next calendar year
(`RwIWENDY`).

### Missing codes

All are Stata extended missing values [15]. Ingestion treats every one as unobserved
(`M = 0`) and tallies them per channel in the ingestion report.

| code | meaning | where |
|---|---|---|
| `.` | did not respond this wave | every wave variable |
| `.D` `.R` | don't know, refused | items |
| `.M` | other missing | items |
| `.Q` | question not asked this wave | e.g. `R2VIGACT` for AHEAD (wave 2A) |
| `.J` | web interview, missing | from wave 11 |
| `.X` | does not apply | `RADAGE_M`: no death date |
| `.U` `.V` | unmarried / spouse did not respond | spouse variables only (none read) |
| `.S` | skip pattern | items |
| `.Y` `.Z` | no exit interview, without / with a death | exit variables only (none read) |
| `.A` | ineligible for the Leave-Behind this wave (the other half) [2068] | `RwLB*` — **not in Table 2** |
| `.N` | Leave-Behind not returned [2078] | `RwLB*` — **Table 2 gives `.N` other meanings elsewhere** |
| `.P` | more than half a scale's items missing [2078] | `RwLB*` — not in Table 2 |
| `.I` | respondent in a nursing home [1373] | `HwINPOV` — not in Table 2 |

### Identity, clock and outcome

| RAND name | meaning | coding | missing | waves | used for |
|---|---|---|---|---|---|
| `HHIDPN` | person id, `1000 × HHID + PN` [13] | numeric | — | — | linking back |
| `HACOHORT` | entry cohort [80] | 0–8, labels above | — | — | report only |
| `RABYEAR` | birth year [120] | year | missing for 1 person; code not shown [121] | — | `S` cohort |
| `RAGENDER` | sex, from the Tracker [142] | `1.Male` `2.Female` | — | — | `S` sex |
| `RADAGE_M` | age at death in months, from `RABDATE`, `RADDATE` [140] | months | `.X` no death date | — | death time |
| `RwIWSTAT` | interview / vital status [74] | `0.Inap.` `1.Resp,alive` `4.NR,alive` `5.NR,died this wv` `6.NR,died prev wv` `7.NR,dropped from samp`; `9` vital status unknown [75] | none | 1–16 | who responded; death after the last interview |
| `RwAGEM_E` | age in months at interview end [140] | months | `.` | 1–16 | the age bin of a wave |
| `RwIWENDY` | interview end year [117] | year | `.` | 1–16 | CPI year for money |
| `RwDIABE` | doctor ever said diabetes or high blood sugar [438] | `0.No` `1.Yes` | `.` only | 1–16 | the disease outcome |

`INWw` (1 if responded) duplicates `RwIWSTAT = 1` and is not read. RAND recommends the
interview-*end* age (`_E`) over begin (`_B`) because most of an interview happens on its
end date [140].

### Chronic conditions (the analysis 04 labels, D23)

Each is "R ever had ...": doctor-diagnosed, `0.No` / `1.Yes` in waves 1–16, with no item
missing codes (only `.` for a wave not responded to). Like `RwDIABE`, disputes rewrite
earlier waves. The label is the flag at the person's interview in the status bin. The
code-side copy is `CONDITIONS` in
[`shared_hrs_chronic.py`](../../src/utils/shared_hrs_chronic.py).

| code | RAND name | condition | codebook page |
|---|---|---|---|
| `hibpe` | `RwHIBPE` | high blood pressure | [425] |
| `diabe` | `RwDIABE` | diabetes | [427] |
| `cancre` | `RwCANCRE` | cancer | [428] |
| `lunge` | `RwLUNGE` | lung disease | [430] |
| `hearte` | `RwHEARTE` | heart problems | [431] |
| `stroke` | `RwSTROKE` | stroke | [433] |
| `psyche` | `RwPSYCHE` | psychiatric problems | [434] |
| `arthre` | `RwARTHRE` | arthritis | [436] |

**Death** is the ninth outcome (D25): death within `death_horizon_years` (default 10)
of the status interview, from `RADAGE_M`. A control must be known alive at the horizon
(a later death date, interview, or `RwIWSTAT = 4` wave). Anyone else is excluded and
counted in the report as `vital_status_unknown_at_horizon`.

### Channels: `C = 12`, four per stream

Order is the row order of `X`. "Into `X`" is what the ingestion writes; everything else
is the variable as stored.

| # | stream | channel | RAND name | meaning | coding | missing | waves | into `X` |
|---|---|---|---|---|---|---|---|---|
| 0 | upstream | income | `HwITOT` | total income of respondent and spouse, **last calendar year**, nominal $ [1353] | ≥ 0, RAND-imputed | `.` only | 1–16 | asinh of thousands of 2022 $, deflated by CPI-U of `RwIWENDY − 1` |
| 1 | upstream | wealth | `HwATOTB` | net worth: all assets incl. second home less all debt, nominal $ [1241] | can be < 0, RAND-imputed | `.` only | 1–16 | asinh of thousands of 2022 $, CPI-U of `RwIWENDY` |
| 2 | upstream | poverty | `HwINPOV` | household income below the poverty threshold [1373] | `0` above, `1` below | `.I` | 1–16 | as is |
| 3 | upstream | working | `RwWORK` | working for pay [1880] | `0` / `1` | `.D` `.R` `.M` `.J` | 1–16 | as is |
| 4 | midstream | loneliness | `RwLBLONELY3` | UCLA loneliness, 3-item mean, higher = lonelier [2078] | 1–3 | `.A` `.N` `.P` `.M` | 8–16 | as is |
| 5 | midstream | life_satisfaction | `RwLBSATWLF` | Diener satisfaction with life, 5-item mean [2111] | **w8: 1–6; w9+: 1–7** | `.A` `.N` `.P` `.M` | 8–16 | w8 mapped `1 + (x − 1)·6/5` |
| 6 | midstream | chronic_stress | `RwLBONCHRSTR` | count of 8 ongoing stressors rated "very upsetting" [2092] | 0–8; item nonresponse counts as 0 | `.A` `.N` `.M` | **8, 10–16** (not asked in 9) | as is |
| 7 | midstream | negative_affect | `RwLBNEGAFFECT` | PANAS-X negative affect, 12-item mean [2097] | 1–5 | `.A` `.N` `.P` `.M` | **9–16** | as is |
| 8 | downstream | bmi | `RwBMI` | self- or proxy-reported BMI; height from waves 1–2 carried forward [577] | kg/m²; range 9.4–102.7 [575] | `.D` `.R` `.M` (assumed; no table printed for continuous variables) | 1–16 | as is, not cleaned |
| 9 | downstream | smoking | `RwSMOKEN` | smokes cigarettes now [626] | `0` / `1` | `.D` `.R` `.M` `.J` | 1–16 | as is |
| 10 | downstream | drinking_days | `RwDRINKD` | days a week with any alcohol, last 3 months [604] | 0–7 (`0` includes non-drinkers) | `.D` `.R` `.M` `.J` | **3–16** | as is |
| 11 | downstream | vigorous_activity | `RwVIGACT` (1–6), `RwVGACTX` (7–16) | vigorous activity 3+ times a week (yes/no) [589] / frequency [593] | `0`/`1`; then `1` every day … `5` never | `.D` `.R` `.M` `.Q`; `.D` `.R` `.M` `.J` | 1–6; 7–16 | 1 if `RwVIGACT = 1` or `RwVGACTX ≤ 2` |

**Why the midstream is all Leave-Behind.** The spec's midstream is observed only in even
bins and for half the people (§1.4), "reproducing the HRS Leave-Behind design". The four
scales above are that design: each household is eligible every other wave from 2006, and
the other half is `.A` [2068]. The core CES-D (`RwCESD`, waves 2–16) would have given
every wave and was rejected for that reason (D15). The cost: no midstream before 2006,
so anyone aged 50–89 only before then has an empty midstream.

### Extended dictionary: every Yan (2026) factor RAND carries (analysis 04, D26)

Yan, *Revisiting the social determinants of health with explainable AI*, AJE 195(3)
(2026), Table S2, lists 61 HRS risk factors from Puterman et al.'s one-row-per-person
file. The ones RAND carries are rebuilt below, wave by wave. The code-side copy, with the
fields of the user's SHARE/ELSA recode records, is `EXTENDED_VARIABLES` and `STATICS` in
[`shared_hrs_variables.py`](../../src/utils/shared_hrs_variables.py). Directions are
natural (higher = more), so the paper's "Lower X" is not reversed. The 12 core channels
above are a subset.

**Not in RAND, so not here:** purpose in life, mastery, optimism, pessimism,
hopelessness, perceived constraints, trait anxiety, anger in/out, cynical hostility,
religiosity, the six positive/negative interaction scales, major and daily
discrimination, adult psychosocial adversity (the nearest RAND measure is chronic stress),
neighbourhood safety/cohesion/disorder, food insecurity, food stamps, father's occupation,
father unemployed, relocation, financial help and adversity in childhood, and the CAGE
alcohol screen. These need the raw core and Leave-Behind files.

**Image rows** (`channel_set: extended`, 30 rows in five domains):

| # | domain | row | RAND | waves | recode | paper factor (Table S2) | in paper's SHARE / ELSA | notes |
|---|---|---|---|---|---|---|---|---|
| 0 | socioeconomic | `income` | `HwITOT` | 1–16 | cpi_real_asinh | Income | ELSA |  |
| 1 | socioeconomic | `wealth` | `HwATOTB` | 1–16 | cpi_real_asinh | Wealth | SHARE/ELSA |  |
| 2 | socioeconomic | `poverty` | `HwINPOV` | 1–16 | direct | — | — |  |
| 3 | socioeconomic | `working` | `RwWORK` | 1–16 | direct | — | — |  |
| 4 | socioeconomic | `unemployed` | `RwUNEMP` | 1–16 | codes_x_a_as_zero | History of Unemployment | SHARE/ELSA | RwUNEMP [1914]; .X not in labour force and .A 65+ presumed retired count as 0, not missing |
| 5 | socioeconomic | `medicaid` | `RwGOVMD` | 1–16 | direct | History of Medicaid | — | RwGOVMD [1615]; the paper's is 'ever', this is per wave |
| 6 | socioeconomic | `not_homeowner` | `HwAHOUS` | 1–16 | zero_value_is_1 | History of Renting | SHARE/ELSA | HwAHOUS = 0 [1203]: renting or otherwise not owning; the household's, as income |
| 7 | socioeconomic | `financial_strain` | `RwLBFINPRB` | 8, 10, 11, 12, 13, 14, 15, 16 | direct | Recent Financial Difficulties | — | RwLBFINPRB [2081], 1 did not happen .. 4 very upsetting; Leave-Behind waves 8, 10-16 (not asked in 9, like chronic stress; found by the schema report 2026-10-25) |
| 8 | adversity | `chronic_stress` | `RwLBONCHRSTR` | 8, 10, 11, 12, 13, 14, 15, 16 | direct | — | — | nearest RAND measure to the paper's adulthood psychosocial adversity; not the same scale |
| 9 | connections | `married_partnered` | `RwMSTAT` | 1–16 | mstat_1_2_3_is_1 | — | — | RwMSTAT [177] currently married, spouse absent or partnered; 0 = separated, divorced, widowed or never married. Not a paper factor (added at the user's request, D27) |
| 10 | connections | `separated_divorced` | `RwMSTAT` | 1–16 | mstat_4_5_6_is_1 | History of Divorce | SHARE/ELSA | RwMSTAT [177] currently separated/divorced; times_divorced carries the history |
| 11 | connections | `times_divorced` | `RwMDIV` | 1–16 | direct | History of Divorce | SHARE/ELSA | RwMDIV [201], 0-6 |
| 12 | psychological | `loneliness` | `RwLBLONELY3` | 8–16 | direct | Loneliness | ELSA |  |
| 13 | psychological | `life_satisfaction` | `RwLBSATWLF` | 8–16 | wave8_rescaled | Lower Life Satisfaction | ELSA |  |
| 14 | psychological | `negative_affect` | `RwLBNEGAFFECT` | 9–16 | direct | Negative Affectivity | SHARE |  |
| 15 | psychological | `positive_affect` | `RwLBPOSAFFECT` | 9–16 | direct | Lower Positive Affectivity | SHARE | RwLBPOSAFFECT [2095], 13 items, 1-5; wave 9 on |
| 16 | psychological | `neuroticism` | `RwLBNEUR` | 8–16 | direct | Lower Neuroticism | — | RwLBNEUR [2070], 1-4 |
| 17 | psychological | `extroversion` | `RwLBEXT` | 8–16 | direct | Lower Extroversion | — | RwLBEXT [2070], 1-4 |
| 18 | psychological | `openness` | `RwLBOPEN` | 8–16 | direct | Lower Openness to Experiences | — | RwLBOPEN [2070], 1-4 |
| 19 | psychological | `agreeableness` | `RwLBAGR` | 8–16 | direct | Lower Agreeableness | — | RwLBAGR [2070], 1-4 |
| 20 | psychological | `conscientiousness` | `RwLBCON5` | 8–16 | direct | Lower Conscientiousness | — | RwLBCON5 [2071], 1-4; the 5-item version spans waves 8-16 |
| 21 | psychological | `depressive_symptoms` | `RwCESD` | 2–16 | direct | — | — | RwCESD [335], 0-8; not a paper factor, the core's only every-wave psychological measure |
| 22 | behaviours | `bmi` | `RwBMI` | 1–16 | direct | — | — |  |
| 23 | behaviours | `smoking` | `RwSMOKEN` | 1–16 | direct | Current Smoker | SHARE/ELSA |  |
| 24 | behaviours | `ever_smoked` | `RwSMOKEV` | 1–16 | direct | History of Smoking | SHARE/ELSA | RwSMOKEV [623] |
| 25 | behaviours | `drinking_days` | `RwDRINKD` | 3–16 | direct | Alcohol Abuse | ELSA | frequency, not the paper's CAGE abuse screen, which RAND lacks |
| 26 | behaviours | `drinks_per_day` | `RwDRINKN` | 3–16 | direct | Alcohol Abuse | ELSA | RwDRINKN [600] |
| 27 | behaviours | `vigorous_activity` | `RwVIGACT, RwVGACTX` | 1–16 | vigact_or_vgactx_le_2 | Low/No Vigorous Activity | SHARE/ELSA |  |
| 28 | behaviours | `moderate_activity` | `RwMDACTX` | 7–16 | le_2_is_1 | Low/No Moderate Activity | SHARE | RwMDACTX [586]; 1 if more than once a week; waves 7-16 only |
| 29 | behaviours | `sleep_problems` | `RwSLEEPFAL` | 6–16 | mean_of_4_items_0_1 | Sleep Problems | SHARE | RwSLEEPFAL/WKN/WKE/RT [497-498]: trouble items 1 most..3 rarely -> (3-x)/2, rested (x-1)/2; mean of 2+ answered; wave 9 and 11 mostly 'answered previous wave' (missing) |

**Static covariates** (`static_set: extended`, 11; a missing value gets the training
mean plus a missing indicator):

| covariate | domain | RAND | paper factor | notes |
|---|---|---|---|---|
| `birth_year` | demographic | `RABYEAR` | — | the cohort covariate (D21) |
| `female` | demographic | `RAGENDER` | Male | RAGENDER = 2 |
| `black` | demographic | `RARACEM` | Black | RARACEM = 2 [144] |
| `other_race` | demographic | `RARACEM` | — | RARACEM = 3 |
| `hispanic` | demographic | `RAHISPAN` | Hispanic | RAHISPAN = 1 [147] |
| `foreign_born` | demographic | `RABPLACE` | Foreign Born | RABPLACE 11 or 13, not US [230]; 12 US territory counts as US |
| `education_years` | socioeconomic | `RAEDYRS` | Lower Education | 0-17 [162] |
| `father_education` | childhood | `RAFEDUC` | Father Education | [171]; 7.5 / 8.5 are 'less than / more than 8'; .D frequent, so a missing indicator goes with it |
| `mother_education` | childhood | `RAMEDUC` | Mother Education | [171] |
| `never_married` | connections | `RwMNEV` | Never Married | RwMNEV at the first interview in the image [200]; .N 'no marriages, likely never married' counts as 1 |
| `lower_occupation` | socioeconomic | `RwJLOCC` | Lower Occupational Status | RwJLOCC 1980 Census [2010]: 5-16 -> 1, 1-4 -> 0, armed forces and other coding (.B, later cohorts) missing |

### Static covariates `S`

| column | from | coding |
|---|---|---|
| `cohort` | `RABYEAR` | z-scored over the people kept |
| `sex` | `RAGENDER` | `1` if `2.Female`, z-scored |
| `frailty` | — | no HRS measure; filled with NaN so a run that asks for it fails visibly |

### Outcome, entry and exclusions

The clock is age at interview end: wave `w` lands in bin `⌊(RwAGEM_E − 600) / 24⌋ + 1`,
bins 1–20 covering ages 50–89.

- **Entry** (`entry_time`) is the bin of the first interview at ages 50–89. Bins before
  it are unobserved by design (left truncation), not by nonresponse.
- **Excluded:** no interview at 50–89; birth year or sex missing; `RwDIABE` missing at
  entry; diabetes reported at or before entry, including interviews before age 50.
- **Disease:** the bin of the first later interview with `RwDIABE = 1`. Onset is between
  that interview and the previous one; the detection bin stands in for it.
- **Death:** the bin of `RADAGE_M`, but only when the last interview was disease-free and
  the next wave records the death (`RwIWSTAT = 5`), or there is no next wave yet. A death
  after a missed interview is censored at the last interview: diabetes could have begun
  in the gap unseen. The report counts these (`deaths_not_counted`).
- **Censored:** at the bin of the last disease-free interview, capped at 20; an event at
  90 or later is censored.
- Nothing after `obs_time` is written into `X`; two waves in one bin keep the later
  non-missing value.

Recorded as D15–D21 in [`decisions_shared.md`](../decisions_shared.md).

## Traps

- **Release letters are not interchangeable.** `h12f3a` and `h14f2b` are different
  releases of different waves; a glob that sorts them lexically will not sort them
  chronologically (`h00…` precedes `h92…`).
- **RAND is derived, the core files are raw.** A variable present in both may not carry
  the same coding, and RAND imputes where the core file leaves a gap. Say which one a
  number came from; this is a property of the *source*, not of any pipeline built on it.
- **Wave-varying availability.** A variable existing in the RAND file does not mean it
  exists in every wave; the prototype's `.dropna()` silently deletes respondents rather
  than reporting the gap. Read the wave column of the dictionary, not a glob.
- **"Ever had" conditions are rewritten by later waves** [438]. A dispute at wave *k*
  sets earlier waves to "no", so `RwDIABE` at wave 5 can depend on what was said at
  wave 9; don't-know and refused are coded `0`, not missing. The two-consecutive-reports
  version `RwDIABE2` [464] needs a confirming later wave, which death and attrition
  remove, so it was not used.
- **Money is nominal**, and `HwITOT` is the *previous* calendar year's income [1353].
  RAND's CPI table ships only as a SAS format [50], so the ingestion carries its own
  CPI-U table, transcribed by hand and checked against the BLS API on 2026-10-25
  (1991–2023 exact; 2024 added).
- **One code, several meanings.** `.N` is "not returned" in the Leave-Behind and nine
  other things elsewhere [15]; `.A`, `.P` and `.I` appear in sections but not in Table 2.
- **The Leave-Behind is not missing at random beyond the half-sample.** 2006 required an
  in-person core interview; from 2016 only non-institutionalised self-respondents
  interviewed in person are eligible [2068]. Return rates among the eligible fell from
  0.89 (2006) to 0.51 (2022) [2065]. New 2022 entrants were never assigned a half.
- **Seams that look like change.** `R8LBSATWLF` is on a 1–6 scale, later waves 1–7
  [2112]; `R8LBNEGAFFECT6` is a different instrument from `RwLBNEGAFFECT` [2097];
  vigorous activity changes question at wave 7 and its wording differs in waves 1–2
  [597]; `R16SMOKEN` has ~2,300 `.M` against ~100 in other waves [626]; from wave 7
  never-smokers skip the question and are coded `0`.
- **BMI is self-reported and uncleaned**: the codebook's range is 9.4–102.7 [575].
- **Dates are coarse.** From wave 3 the interview day is imputed as the 15th [117];
  1,321 death dates are imputed (`RADSRC = 4`) [128].
- **`RwIWSTAT = 9`**: vital status unknown [75]. It is neither a response nor a death.

## Settling the tier

To move off `blind`, find and quote here the clause of the HRS / RAND data use agreement
that governs transfer of data to a third party, then set the tier accordingly. The
codebook names the agreement: access requires registering with HRS, which means agreeing
to its "Conditions of Use" at <https://hrsdata.isr.umich.edu/data-products>, and the same
agreement covers the RAND products [13]. That document is where the clause will be. Two
things follow, and they are the reason this matters more than the paperwork suggests:

1. **Contract before policy.** If the agreement forbids transfer outright, no vendor
   retention or training policy answers it. Check what the agreement permits before
   checking what any platform permits.
2. **A `blind` source needs a fixture or the tier defeats itself.** One now exists:
   [`tests/fixtures/rand_hrs_fixture.py`](../../tests/fixtures/rand_hrs_fixture.py)
   writes a Stata file with the dictionary's names, labels, wave gaps and extended
   missing codes. It covers only the columns above; a new variable needs adding there
   first. It is built from the codebook, so it cannot catch the codebook being wrong —
   the first real run should be read as a test of the fixture too.
