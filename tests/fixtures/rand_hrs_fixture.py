"""Synthetic RAND HRS Longitudinal File: the real file's schema and quirks, none of its rows.

HRS is ``blind`` (D07): no assistant may read the real ``.dta``. This module writes a
small Stata file that the ingestion code (``utils.shared_hrs_ingest``) can be developed
and tested against. Everything in it comes from the codebook,
``data/hrs/randhrs1992_2022v1_STATA/randhrs1992_2022v1.pdf``, never from the data.

Run it directly to write a file to look at::

    python tests/fixtures/rand_hrs_fixture.py [out.dta] [--uppercase]

The tests call :func:`write_fixture` into a temporary directory instead.

**What it reproduces** (codebook page in brackets):

* Names, wave availability and value labels of every column the dictionary in
  ``doc/data/hrs.md`` uses, plus ``r8lbnegaffect6``, a wave-8 scale on a different
  instrument that must not be mistaken for ``r8lbnegaffect`` (which does not exist).
* Stata extended missing codes, written as real ``.d``/``.a``/... bytes: Table 2's
  ``. .d .r .m .q .j .x`` [15] and the section-specific ``.a`` ineligible for the
  Leave-Behind, ``.n`` LB not returned, ``.p`` partial LB scale [2078] and ``.i``
  in a nursing home for ``HwINPOV`` [1373].
* Staggered entry: eight cohorts entering in 1992, 1993, 1998, 2004, 2010, 2016 and
  2022, with AHEAD's waves 2-3 fielded in 1993/1995 [13], spouses outside the
  cohort's birth years (some under 50 at entry) and fieldwork spilling into the
  next calendar year.
* Variables missing in some waves: Leave-Behind scales from wave 8, chronic stress
  absent in wave 9, negative affect from wave 9, drinking days from wave 3, the
  vigorous-activity break between ``RwVIGACT`` (1-6) and ``RwVGACTX`` (7-16), and
  ``.q`` for AHEAD's wave-2 activity questions [596].
* The Leave-Behind half-sample: each household is assigned a half, eligible every
  other wave from 2006; the rest are ``.a``. New 2022 cohort members are not
  assigned a half and are never ``.a`` [2068].
* Death and attrition through ``RwIWSTAT``: 4 alive non-response (temporary), 7
  dropped, 9 vital status unknown, 5 died since the last wave, 6 died earlier [75];
  ``RADAGE_M`` is ``.x`` without a death date [140], and deaths can follow attrition.

**What it does not.** Storage widths (byte/int/long/float/double) are the fixture's
choice: the codebook gives only Cont/Categ. Missing-code *rates* are inflated so
that every code occurs in a few hundred rows; the values are plausible, not
calibrated. RAND also labels its missing codes (".D=DK"); pandas cannot write labels
for extended missing values, so the fixture leaves them unlabelled. The text of the
``RwIWSTAT`` label for code 9 is not shown in the codebook; the fixture's is a
placeholder.
"""

from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FIXTURE_SEED = 20260724
N_HOUSEHOLDS = 300
DEFAULT_PATH = Path(__file__).with_name("randhrs_fixture.dta")
N_WAVES = 16
LAST_DEATH_YEAR = 2023  # RADYEAR runs to 2023 [5312]


def _waves(first: int, last: int = N_WAVES) -> tuple[int, ...]:
    return tuple(range(first, last + 1))


ALL = _waves(1)
# Wave availability, from the codebook's variable listings. Kept apart from the
# ingestion module's own table on purpose: the tests compare the two.
COLUMN_WAVES: dict[str, tuple[int, ...]] = {
    "inw{w}": ALL,
    "r{w}iwstat": ALL,
    "r{w}agem_e": ALL,
    "r{w}iwendy": ALL,
    "r{w}diabe": ALL,
    # The other seven "ever had" conditions, all 16 waves, 0/1 [425-436].
    "r{w}hibpe": ALL,
    "r{w}cancre": ALL,
    "r{w}lunge": ALL,
    "r{w}hearte": ALL,
    "r{w}stroke": ALL,
    "r{w}psyche": ALL,
    "r{w}arthre": ALL,
    "h{w}itot": ALL,
    "h{w}atotb": ALL,
    "h{w}inpov": ALL,
    "r{w}work": ALL,
    "r{w}lblonely3": _waves(8),
    "r{w}lbsatwlf": _waves(8),
    "r{w}lbonchrstr": (8,) + _waves(10),
    "r{w}lbnegaffect": _waves(9),
    "r{w}lbnegaffect6": (8,),
    "r{w}bmi": ALL,
    "r{w}smoken": ALL,
    "r{w}drinkd": _waves(3),
    "r{w}vigact": _waves(1, 6),
    "r{w}vgactx": _waves(7),
    # The extended dictionary (shared_hrs_variables, D26).
    "r{w}unemp": ALL,
    "r{w}govmd": ALL,
    "h{w}ahous": ALL,
    "r{w}lbfinprb": (8,) + _waves(10),  # not asked in wave 9, as chronic stress
    "r{w}mstat": ALL,
    "r{w}mdiv": ALL,
    "r{w}mnev": ALL,
    "r{w}jlocc": ALL,
    "r{w}lbposaffect": _waves(9),
    "r{w}lbneur": _waves(8),
    "r{w}lbext": _waves(8),
    "r{w}lbopen": _waves(8),
    "r{w}lbagr": _waves(8),
    "r{w}lbcon5": _waves(8),
    "r{w}cesd": _waves(2),
    "r{w}smokev": ALL,
    "r{w}drinkn": _waves(3),
    "r{w}mdactx": _waves(7),
    "r{w}sleepfal": _waves(6),
    "r{w}sleepwkn": _waves(6),
    "r{w}sleepwke": _waves(6),
    "r{w}sleeprt": _waves(6),
}
STATIC = (
    "hhidpn", "hacohort", "rabyear", "rabmonth", "ragender", "radage_m", "radyear",
    "raracem", "rahispan", "rabplace", "raedyrs", "rafeduc", "rameduc",
)

STORAGE: dict[str, type] = {
    "hhidpn": np.int32, "hacohort": np.int8, "rabyear": np.int16, "rabmonth": np.int8,
    "ragender": np.int8, "radage_m": np.int16, "radyear": np.int16,
    "inw{w}": np.int8, "r{w}iwstat": np.int8, "r{w}agem_e": np.int16, "r{w}iwendy": np.int16,
    "r{w}diabe": np.int8, "r{w}hibpe": np.int8, "r{w}cancre": np.int8, "r{w}lunge": np.int8,
    "r{w}hearte": np.int8, "r{w}stroke": np.int8, "r{w}psyche": np.int8, "r{w}arthre": np.int8,
    "h{w}itot": np.float64, "h{w}atotb": np.float64, "h{w}inpov": np.int8,
    "r{w}work": np.int8, "r{w}lblonely3": np.float64, "r{w}lbsatwlf": np.float64,
    "r{w}lbonchrstr": np.float64, "r{w}lbnegaffect": np.float64, "r{w}lbnegaffect6": np.float64,
    "r{w}bmi": np.float32, "r{w}smoken": np.int8, "r{w}drinkd": np.int8, "r{w}vigact": np.int8,
    "r{w}vgactx": np.int8,
    "raracem": np.int8, "rahispan": np.int8, "rabplace": np.int8, "raedyrs": np.int8,
    "rafeduc": np.float32, "rameduc": np.float32,  # 7.5 / 8.5 are codes [171]
    "r{w}unemp": np.int8, "r{w}govmd": np.int8, "h{w}ahous": np.float64, "r{w}lbfinprb": np.int8,
    "r{w}mstat": np.int8, "r{w}mdiv": np.int8, "r{w}mnev": np.int8, "r{w}jlocc": np.int8,
    "r{w}lbposaffect": np.float64, "r{w}lbneur": np.float64, "r{w}lbext": np.float64, "r{w}lbopen": np.float64,
    "r{w}lbagr": np.float64, "r{w}lbcon5": np.float64, "r{w}cesd": np.int8, "r{w}smokev": np.int8,
    "r{w}drinkn": np.int8, "r{w}mdactx": np.int8, "r{w}sleepfal": np.int8, "r{w}sleepwkn": np.int8,
    "r{w}sleepwke": np.int8, "r{w}sleeprt": np.int8,
}

_NO_YES = {0: "0.No", 1: "1.Yes"}
VALUE_LABELS: dict[str, dict[int, str]] = {
    "hacohort": {
        0: "0.Hrs/Ahead ovrlap", 1: "1.Ahead", 2: "2.Coda", 3: "3.Hrs", 4: "4.WarBabies",
        5: "5.Early BabyBoomers", 6: "6.Mid BabyBoomers", 7: "7.Late BabyBoomers",
        8: "8.Early Generation X",
    },
    "ragender": {1: "1.Male", 2: "2.Female"},
    "inw{w}": {0: "0.NonResp", 1: "1.Resp,alive"},
    "r{w}iwstat": {
        0: "0.Inap.", 1: "1.Resp,alive", 4: "4.NR,alive", 5: "5.NR,died this wv",
        6: "6.NR,died prev wv", 7: "7.NR,dropped from samp", 9: "9.NR,dk if alive",
    },
    "r{w}diabe": _NO_YES,
    **{f"r{{w}}{c}": _NO_YES for c in ("hibpe", "cancre", "lunge", "hearte", "stroke", "psyche", "arthre")},
    "h{w}inpov": {0: "0.HH inc above pov thresh", 1: "1.HH inc below pov thresh"},
    "r{w}work": {0: "0.Not working for pay", 1: "1.Working for pay"},
    "r{w}smoken": _NO_YES,
    "r{w}drinkd": {0: "0.0 or doesnt drink"},
    "r{w}vigact": _NO_YES,
    "r{w}vgactx": {1: "1.Every day", 2: "2.>1 per week", 3: "3.1 per week", 4: "4.l-3 per mon", 5: "5.Never"},
    "r{w}mdactx": {1: "1.Every day", 2: "2.>1 per week", 3: "3.1 per week", 4: "4.l-3 per mon", 5: "5.Never"},
    "raracem": {1: "1.White/Caucasian", 2: "2.Black/African American", 3: "3.Other"},
    "rahispan": {0: "0.Not Hispanic", 1: "1.Hispanic"},
    "rabplace": {1: "1.New England", 5: "5.S Atlantic", 9: "9.Pacific", 11: "11.Not US or US terr", 12: "12.US territory", 13: "13.Not US"},
    "r{w}unemp": _NO_YES, "r{w}govmd": _NO_YES, "r{w}smokev": _NO_YES,
    "r{w}mnev": {0: "0.Ever married", 1: "1.Never married"},
    "r{w}mstat": {1: "1.Married", 2: "2.Married,spouse absent", 3: "3.Partnered", 4: "4.Separated", 5: "5.Divorced",
                  6: "6.Separated/divorced", 7: "7.Widowed", 8: "8.Never married"},
    "r{w}lbfinprb": {1: "1. No did not happen", 2: "2. Yes but not upsetting", 3: "3. Yes somewhat upsetting",
                     4: "4. Yes very upsetting"},
    **{f"r{{w}}sleep{k}": {1: "1.Most of the time", 2: "2.Sometimes", 3: "3.Rarely or never"} for k in ("fal", "wkn", "wke", "rt")},
}

VARIABLE_LABELS: dict[str, str] = {
    "hhidpn": "HHold ID + Person Number /Num", "hacohort": "Sample cohort",
    "rabyear": "R birth year", "rabmonth": "R birth month", "ragender": "R Gender",
    "radage_m": "Age at Death in Months", "radyear": "R death year",
    "inw{w}": "=1 if Respondent", "r{w}iwstat": "R Interview Status",
    "r{w}agem_e": "R age (months) at ivw end", "r{w}iwendy": "Interview End Year",
    "r{w}diabe": "R ever had diabetes", "r{w}hibpe": "R ever had high blood pressure",
    "r{w}cancre": "R ever had cancer", "r{w}lunge": "R ever had lung disease",
    "r{w}hearte": "R ever had heart problems", "r{w}stroke": "R ever had stroke",
    "r{w}psyche": "R ever had psych problems", "r{w}arthre": "R ever had arthritis",
    "h{w}itot": "Total HHold / R+Sp only",
    "h{w}atotb": "Total all Assets inc. 2nd Hm", "h{w}inpov": "Whether in Poverty",
    "r{w}work": "R working for pay", "r{w}lblonely3": "Loneliness 3-item scale",
    "r{w}lbsatwlf": "Life satisfaction scale", "r{w}lbonchrstr": "Ongoing Chronic Stress Scale",
    "r{w}lbnegaffect": "Negative affect 12-question scale",
    "r{w}lbnegaffect6": "Negative affect 6-question scale", "r{w}bmi": "R Body mass index",
    "r{w}smoken": "R smokes now", "r{w}drinkd": "R # days/week drinks",
    "r{w}vigact": "R Wtr vigorus phys act 3+/wk", "r{w}vgactx": "R Freq vigorous phys activ",
    "raracem": "R Race-masked", "rahispan": "R Hispanic", "rabplace": "R Place of Birth (Cens Region)",
    "raedyrs": "R Years of Education", "rafeduc": "R Father's Years Education", "rameduc": "R Mother's Years Education",
    "r{w}unemp": "=1 if R is unemployed", "r{w}govmd": "R has Gov plan-Medicaid",
    "h{w}ahous": "Assets:Primary Residence--Cross-wave", "r{w}lbfinprb": "Ongoing financial strain",
    "r{w}mstat": "R Marital Status", "r{w}mdiv": "R # times divorced", "r{w}mnev": "R never married",
    "r{w}jlocc": "Longest Job Occ/1980 Census", "r{w}lbposaffect": "Positive affect 13-question scale",
    "r{w}lbneur": "Big 5 Neuroticism", "r{w}lbext": "Big 5 Extroversion", "r{w}lbopen": "Big 5 Openness to experience",
    "r{w}lbagr": "Big 5 Agreeableness", "r{w}lbcon5": "Big 5 Conscientiousness 5 Sub-items", "r{w}cesd": "CESD score",
    "r{w}smokev": "R smoke ever", "r{w}drinkn": "R # drinks/day when drinks",
    "r{w}mdactx": "R Freq moderate phys activ {finer scale}",
    "r{w}sleepfal": "R reports trouble falling asleep", "r{w}sleepwkn": "R reports trouble waking up during night",
    "r{w}sleepwke": "R reports trouble waking up too early", "r{w}sleeprt": "R reports feeling rested in AM",
}

# The other "ever had" conditions: (log hazard per year from age 35, SES slope). Rough
# prevalences, high blood pressure and arthritis common, stroke rare; not calibrated.
OTHER_CONDITIONS = {
    "hibpe": (-3.3, -0.3), "cancre": (-4.6, 0.1), "lunge": (-4.8, -0.3), "hearte": (-4.2, -0.3),
    "stroke": (-5.2, -0.3), "psyche": (-4.6, -0.2), "arthre": (-3.4, -0.2),
}

# hacohort: (first birth year, last birth year, entry wave, sampling weight) [12, 81]
COHORTS = {
    0: (1915, 1923, 1, 0.03), 1: (1905, 1923, 2, 0.14), 2: (1924, 1930, 4, 0.08),
    3: (1931, 1941, 1, 0.25), 4: (1942, 1947, 4, 0.10), 5: (1948, 1953, 7, 0.10),
    6: (1954, 1959, 10, 0.10), 7: (1960, 1965, 13, 0.10), 8: (1966, 1971, 16, 0.10),
}
AHEAD_TIMING = (0, 1)  # waves 2 and 3 fielded in 1993 and 1995, not 1994 and 1996

# A linear stand-in for CPI-U, only so the nominal dollar series drift upward.
_CPI = dict(zip(range(1990, 2025), np.linspace(132.0, 310.0, 35)))


def wave_year(wave: int, cohort: int) -> int:
    if wave == 1:
        return 1992
    if wave in (2, 3):
        return (1993 if wave == 2 else 1995) if cohort in AHEAD_TIMING else 1990 + 2 * wave
    return 1998 + 2 * (wave - 4)


@dataclass(frozen=True)
class Fixture:
    """Every column as (values, codes); a code is ``""`` where the value is valid."""

    columns: dict[str, tuple[np.ndarray, np.ndarray]]
    storage: dict[str, type]
    value_labels: dict[str, dict[int, str]]
    variable_labels: dict[str, str]
    people: pd.DataFrame  # the latent facts each row was generated from

    @property
    def n(self) -> int:
        return len(self.people)


def _family(column: str) -> str:
    """``r12bmi`` -> ``r{w}bmi``; a static name maps to itself."""
    if column in STATIC:
        return column
    for stem in COLUMN_WAVES:
        head, tail = stem.split("{w}")
        middle = column[len(head): len(column) - len(tail)]
        if column.startswith(head) and column.endswith(tail) and middle.isdigit():
            return stem
    raise KeyError(column)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def build_fixture(n_households: int = N_HOUSEHOLDS, seed: int = FIXTURE_SEED) -> Fixture:
    rng = np.random.default_rng(seed)
    people = _people(rng, n_households)
    dates = _interview_dates(rng, people)
    _add_death(rng, people, dates)
    status = _interview_status(rng, people, dates)
    n = len(people)
    columns: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def put(name: str, values: np.ndarray, codes: np.ndarray | None = None) -> None:
        values = np.asarray(values, dtype=float)
        if codes is None:
            codes = np.where(np.isnan(values), ".", "").astype(object)
        columns[name] = (np.where(codes == "", values, np.nan), codes)

    put("hhidpn", people.hhidpn)
    put("hacohort", people.hacohort)
    put("rabyear", people.birth_year)
    put("rabmonth", people.birth_month)
    put("ragender", people.gender)
    died = people.death_m.notna() & (people.death_year <= LAST_DEATH_YEAR)
    put("radage_m", np.where(died, people.death_m, np.nan), np.where(died, "", ".x").astype(object))
    put("radyear", np.where(died, people.death_year, np.nan), np.where(died, "", ".x").astype(object))
    n_people = len(people)
    z0 = people.ses.to_numpy()

    def static_codes(rate: float, code: str = ".m") -> np.ndarray:
        return np.where(rng.random(n_people) < rate, code, "").astype(object)

    put("raracem", rng.choice([1, 2, 3], size=n_people, p=[0.72, 0.2, 0.08]), static_codes(0.01))
    put("rahispan", (rng.random(n_people) < 0.13).astype(float), static_codes(0.01))
    put("rabplace", rng.choice([1, 5, 9, 11, 12, 13], size=n_people, p=[0.2, 0.3, 0.33, 0.03, 0.01, 0.13]), static_codes(0.01))
    put("raedyrs", np.clip(np.round(12 + 2.5 * z0 + rng.normal(0, 1.5, n_people)), 0, 17), static_codes(0.015))
    for column in ("rafeduc", "rameduc"):
        years = np.clip(np.round(9 + 2.5 * z0 + rng.normal(0, 2.5, n_people)), 0, 17)
        years = np.where(rng.random(n_people) < 0.05, rng.choice([7.5, 8.5], size=n_people), years)
        put(column, years, np.where(rng.random(n_people) < 0.12, ".d", static_codes(0.02)))
    never_married = rng.random(n_people) < 0.04
    occupation = np.where(z0 > 0, rng.integers(1, 5, n_people), rng.integers(5, 17, n_people))
    later_coding = people.hacohort.to_numpy() >= 6  # later cohorts' longest job is in another Census coding

    birth_ym =people.birth_year.to_numpy() * 12 + people.birth_month.to_numpy() - 1
    z = people.ses.to_numpy()
    half = people.lb_half.to_numpy()
    cohort = people.hacohort.to_numpy()
    for w in ALL:
        responded = status[:, w - 1] == 1
        year, month = dates[:, w - 1, 0], dates[:, w - 1, 1]
        age_m = np.where(responded, year * 12 + month - 1 - birth_ym, np.nan)
        age = age_m / 12.0
        put(f"inw{w}", responded.astype(float))
        put(f"r{w}iwstat", status[:, w - 1])
        put(f"r{w}agem_e", age_m)
        put(f"r{w}iwendy", np.where(responded, year, np.nan))
        put(f"r{w}diabe", np.where(responded, (people.onset_m.to_numpy() <= age_m).astype(float), np.nan))
        for condition in OTHER_CONDITIONS:
            onset = people[f"onset_m_{condition}"].to_numpy()
            put(f"r{w}{condition}", np.where(responded, (onset <= age_m).astype(float), np.nan))

        def coded(values: np.ndarray, rates: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
            """Respondent cells take a code at the given rates; others are plain '.'."""
            codes = np.where(responded, "", ".").astype(object)
            draw = rng.random(n)
            edge = 0.0
            for code, rate in rates.items():
                hit = responded & (draw >= edge) & (draw < edge + rate)
                codes[hit] = code
                edge += rate
            return values, codes

        noise = rng.normal(size=(8, n))
        price = np.array([_CPI[int(y)] for y in year])
        price_last_year = np.array([_CPI[int(y) - 1] for y in year])
        income = np.exp(10.4 + 0.5 * z + 0.4 * noise[0]) * np.where(rng.random(n) < 0.02, 0.0, 1.0)
        put(f"h{w}itot", *coded(income * price_last_year / _CPI[1991], {}))
        wealth = np.exp(11.3 + 1.0 * z + 0.8 * noise[1]) - 60000.0 * (rng.random(n) < 0.1)
        put(f"h{w}atotb", *coded(wealth * price / _CPI[1992], {}))
        put(f"h{w}inpov", *coded((income < 12000).astype(float), {".i": 0.03}))
        work = rng.random(n) < _sigmoid(2.5 - 0.25 * (age - 60) + 0.5 * z)
        put(f"r{w}work", *coded(work.astype(float), {".d": 0.01, ".r": 0.01, ".m": 0.02} | ({".j": 0.01} if w >= 11 else {})))

        if w >= 8:
            # Eligible in alternate waves by household half; the 2022 entrants were
            # never assigned one, so they are never ".a" [2068].
            new_2022 = (cohort == 8) & (w == 16)
            eligible = ((w - 8) % 2 == half) | new_2022
            lonely = np.clip(np.round((1.5 - 0.3 * z + 0.5 * noise[2]) * 3) / 3, 1, 3)
            top = 6 if w == 8 else 7
            satisfied = np.clip(np.round((top * 0.7 + 0.5 * z + noise[3]) * 5) / 5, 1, top)
            stress = rng.binomial(8, _sigmoid(-2.5 - 0.5 * z)).astype(float)
            negative = np.clip(np.round((1.7 - 0.2 * z + 0.6 * noise[4]) * 12) / 12, 1, 5)
            returned = rng.random(n)
            big5 = {k: np.clip(np.round((2.8 + 0.4 * rng.normal(size=n)) * 4) / 4, 1, 4)
                    for k in ("lbneur", "lbext", "lbopen", "lbagr", "lbcon5")}
            for stem, values in (
                ("r{w}lblonely3", lonely), ("r{w}lbsatwlf", satisfied),
                ("r{w}lbonchrstr", stress), ("r{w}lbnegaffect", negative), ("r{w}lbnegaffect6", negative),
                ("r{w}lbfinprb", rng.integers(1, 5, n).astype(float)),
                ("r{w}lbposaffect", np.clip(np.round((3.4 + 0.2 * z + 0.6 * noise[3]) * 13) / 13, 1, 5)),
                *((f"r{{w}}{k}", v) for k, v in big5.items()),
            ):
                if w not in COLUMN_WAVES[stem]:
                    continue
                values, codes = coded(values, {})
                codes[responded & ~eligible] = ".a"
                ok = responded & eligible
                codes[ok & (returned < 0.20)] = np.where(new_2022[ok & (returned < 0.20)], ".m", ".n")
                codes[ok & (returned >= 0.20) & (returned < 0.23)] = ".p"
                codes[ok & (returned >= 0.23) & (returned < 0.25)] = ".m"
                put(stem.format(w=w), values, codes)

        bmi = np.clip(27 - 0.8 * z + 4 * noise[5], 14, 60)
        bmi = np.where(rng.random(n) < 0.01, np.where(rng.random(n) < 0.5, 9.4, 95.0), bmi)  # [28465]
        put(f"r{w}bmi", *coded(np.round(bmi, 1), {".d": 0.01, ".r": 0.01, ".m": 0.01}))
        smokes = rng.random(n) < _sigmoid(-1.2 - 0.6 * z - 0.03 * (age - 60))
        put(f"r{w}smoken", *coded(smokes.astype(float), {".d": 0.005, ".m": 0.15 if w == 16 else 0.01}))
        # The extended dictionary's every-wave items.
        employed_age = responded & (age < 65)
        unemployed = (rng.random(n) < 0.05).astype(float)
        unemp_codes = np.where(responded, np.where(employed_age, "", ".x"), ".").astype(object)
        if w == 1:
            unemp_codes[responded & (age >= 65)] = ".a"
        put(f"r{w}unemp", unemployed, unemp_codes)
        put(f"r{w}govmd", *coded((rng.random(n) < _sigmoid(-2.5 - 0.8 * z)).astype(float), {".d": 0.01, ".m": 0.01}))
        owns = rng.random(n) < _sigmoid(1.2 + 0.8 * z)
        put(f"h{w}ahous", *coded(np.where(owns, np.exp(11.8 + 0.6 * z + 0.4 * noise[7]), 0.0), {}))
        marital = np.where(never_married, 8, rng.choice([1, 3, 5, 7], size=n, p=[0.65, 0.05, 0.15, 0.15]))
        put(f"r{w}mstat", *coded(marital.astype(float), {".m": 0.005}))
        put(f"r{w}mdiv", *coded(np.where(never_married, 0, rng.poisson(0.4, n)).astype(float), {}))
        mnev_codes = np.where(responded, "", ".").astype(object)
        mnev_codes[responded & never_married & (rng.random(n) < 0.1)] = ".n"
        put(f"r{w}mnev", never_married.astype(float), mnev_codes)
        jlocc_codes = np.where(responded, np.where(later_coding & (w >= 10), ".b", ""), ".").astype(object)
        jlocc_codes[(jlocc_codes == "") & responded & (rng.random(n) < 0.08)] = ".m"
        put(f"r{w}jlocc", occupation.astype(float), jlocc_codes)
        put(f"r{w}smokev", *coded((rng.random(n) < 0.55).astype(float), {".m": 0.01}))
        if w >= 2:
            put(f"r{w}cesd", *coded(rng.binomial(8, _sigmoid(-1.8 - 0.4 * z)).astype(float), {".m": 0.03}))
        if w >= 6:
            sleep_rates = {".d": 0.01, ".r": 0.005} | ({".p": 0.5} if w in (9, 11) else {})  # w9/w11: answered before
            for item in ("fal", "wkn", "wke", "rt"):
                put(f"r{w}sleep{item}", *coded(rng.integers(1, 4, n).astype(float), sleep_rates))
        if w >= 7:
            put(f"r{w}mdactx", *coded(rng.integers(1, 6, n).astype(float), {".d": 0.01, ".r": 0.005}))
        if w >= 3:
            put(f"r{w}drinkn", *coded(rng.integers(0, 5, n).astype(float), {".d": 0.01, ".m": 0.01}))
        if w >= 3:
            days = np.where(rng.random(n) < 0.55, 0, rng.integers(1, 8, n)).astype(float)
            put(f"r{w}drinkd", *coded(days, {".d": 0.01, ".r": 0.01, ".m": 0.01} | ({".j": 0.01} if w >= 11 else {})))
        active = _sigmoid(-0.5 + 0.4 * z - 0.03 * (age - 60))
        if w <= 6:
            values, codes = coded((rng.random(n) < active).astype(float), {".d": 0.01, ".r": 0.01, ".m": 0.01})
            if w == 2:
                codes[responded & np.isin(cohort, AHEAD_TIMING)] = ".q"  # not asked in 2A [596]
            put(f"r{w}vigact", values, codes)
        else:
            frequency = np.clip(np.round(3.6 - 1.2 * (active - 0.4) * 3 + noise[6]), 1, 5)
            put(f"r{w}vgactx", *coded(frequency, {".d": 0.01, ".r": 0.01} | ({".j": 0.01} if w >= 11 else {})))

    ordered = [s for s in STATIC] + [stem.format(w=w) for stem, waves in COLUMN_WAVES.items() for w in waves]
    columns = {name: columns[name] for name in ordered}
    storage = {name: STORAGE[_family(name)] for name in columns}
    value_labels = {name: VALUE_LABELS[_family(name)] for name in columns if _family(name) in VALUE_LABELS}
    # RAND's label pattern: "NAME:Ww concept" for wave variables, "NAME: concept" otherwise [14].
    variable_labels = {}
    for name in columns:
        family = _family(name)
        text = VARIABLE_LABELS[family]
        wave = "" if family == name else f"W{_wave_of(name, family)}"
        variable_labels[name] = f"{name.upper()}:{wave} {text}"
    return Fixture(columns, storage, value_labels, variable_labels, people)


def _wave_of(name: str, family: str) -> int:
    head, tail = family.split("{w}")
    return int(name[len(head): len(name) - len(tail)])


def _people(rng: np.random.Generator, n_households: int) -> pd.DataFrame:
    """One row per person: cohort, birth, latent SES and true diabetes onset age."""
    codes = np.array(list(COHORTS))
    weights = np.array([COHORTS[c][3] for c in codes])
    rows = []
    for h in range(n_households):
        cohort = int(rng.choice(codes, p=weights / weights.sum()))
        first, last, entry_wave, _ = COHORTS[cohort]
        hhid = 10000 + 7 * h
        half = int(rng.integers(2))
        members = [int(rng.integers(first, last + 1))]
        if rng.random() < 0.45:  # a spouse, who need not be age-eligible [12]
            members.append(members[0] + int(rng.integers(-6, 13)))
        for k, birth_year in enumerate(members):
            rows.append(
                dict(
                    hhidpn=hhid * 1000 + 10 * (k + 1), hacohort=cohort, entry_wave=entry_wave,
                    birth_year=birth_year, birth_month=int(rng.integers(1, 13)),
                    gender=int(rng.integers(1, 3)), ses=float(rng.normal()), lb_half=half,
                )
            )
    people = pd.DataFrame(rows)
    z = people.ses.to_numpy()
    # Diabetes: constant hazard from age 35, higher at low SES; some prevalent at entry.
    people["onset_m"] = np.floor((35 + rng.exponential(1.0 / np.exp(-4.4 - 0.3 * z))) * 12)
    for condition, (log_hazard, slope) in OTHER_CONDITIONS.items():
        people[f"onset_m_{condition}"] = np.floor((35 + rng.exponential(1.0 / np.exp(log_hazard + slope * z))) * 12)
    return people


def _interview_dates(rng: np.random.Generator, people: pd.DataFrame) -> np.ndarray:
    """(n, waves, 2) interview (year, month); households interview together."""
    dates = np.zeros((len(people), N_WAVES, 2), dtype=int)
    for hhid, rows in people.groupby(people.hhidpn // 1000).groups.items():
        cohort = int(people.hacohort[rows[0]])
        for w in ALL:
            year, month = wave_year(w, cohort), int(rng.integers(2, 13))
            if rng.random() < 0.08:  # fieldwork runs into the next calendar year
                year, month = year + 1, int(rng.integers(1, 4))
            dates[rows, w - 1] = (year, month)
    return dates


def _add_death(rng: np.random.Generator, people: pd.DataFrame, dates: np.ndarray) -> None:
    """Gompertz death age, drawn beyond the entry interview so everyone has one."""
    birth_ym = (people.birth_year * 12 + people.birth_month - 1).to_numpy()
    entry = dates[np.arange(len(people)), people.entry_wave.to_numpy() - 1]
    entry_age = (entry[:, 0] * 12 + entry[:, 1] - birth_ym) / 12.0
    b, c = np.exp(-10.3 - 0.35 * people.ses.to_numpy()), 0.09
    death = np.log(np.exp(c * entry_age) - c * np.log(rng.random(len(people))) / b) / c
    people["death_m"] = np.floor(death * 12)
    people["death_year"] = ((birth_ym + people.death_m) // 12).astype(int)


def _interview_status(rng: np.random.Generator, people: pd.DataFrame, dates: np.ndarray) -> np.ndarray:
    """``RwIWSTAT`` per person and wave."""
    status = np.zeros((len(people), N_WAVES), dtype=int)
    birth_ym = (people.birth_year * 12 + people.birth_month - 1).to_numpy()
    death_ym = birth_ym + people.death_m.to_numpy()
    for i, person in enumerate(people.itertuples()):
        dropped = dead_reported = False
        for w in ALL:
            year, month = dates[i, w - 1]
            if w < person.entry_wave:
                continue  # 0: not yet in the sample
            if year * 12 + month - 1 > death_ym[i]:
                status[i, w - 1] = 6 if dead_reported else 5
                dead_reported = True
            elif w == person.entry_wave:
                status[i, w - 1] = 1
            elif dropped or rng.random() < 0.03:
                dropped = True
                status[i, w - 1] = 9 if rng.random() < 0.1 else 7
            else:
                status[i, w - 1] = 4 if rng.random() < 0.06 else 1
    return status


# --- writing --------------------------------------------------------------------
_TYPE_WIDTH = {65530: 1, 65529: 2, 65528: 4, 65527: 4, 65526: 8}  # byte int long float double


def _missing_bytes(stata_type: int, code: str) -> bytes:
    """Stata's little-endian encoding of ``.``, ``.a`` ... ``.z`` for a storage type."""
    k = 0 if code == "." else ord(code[1]) - 96
    if stata_type == 65530:
        return struct.pack("<b", 101 + k)
    if stata_type == 65529:
        return struct.pack("<h", 32741 + k)
    if stata_type == 65528:
        return struct.pack("<i", 2147483621 + k)
    if stata_type == 65527:
        return struct.pack("<I", 0x7F000000 + k * 0x800)
    if stata_type == 65526:
        return struct.pack("<Q", 0x7FE0000000000000 + k * 0x10000000000)
    raise ValueError(f"no missing values for Stata type {stata_type}")


def write_fixture(path: Path = DEFAULT_PATH, fixture: Fixture | None = None, uppercase: bool = False) -> Path:
    """Write ``fixture`` as a Stata 118 ``.dta`` with its missing codes in place.

    pandas writes only system missing (``.``), so cells carrying a code are written
    as 0 and then overwritten in the data section with the code's bytes.
    """
    fixture = fixture or build_fixture()
    rename = (lambda c: c.upper()) if uppercase else (lambda c: c)
    frame = pd.DataFrame(
        {
            rename(name): np.where(codes == "", np.nan_to_num(values), 0).astype(fixture.storage[name])
            for name, (values, codes) in fixture.columns.items()
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_stata(
        path,
        write_index=False,
        version=118,
        byteorder="<",
        variable_labels={rename(k): v for k, v in fixture.variable_labels.items()},
        value_labels={rename(k): v for k, v in fixture.value_labels.items()},
    )
    _patch_missing(path, [codes for _, codes in fixture.columns.values()])
    return path


def _patch_missing(path: Path, codes: list[np.ndarray]) -> None:
    raw = bytearray(path.read_bytes())
    at = raw.index(b"<map>") + len(b"<map>")
    offsets = struct.unpack("<14Q", raw[at: at + 14 * 8])
    types_at = offsets[2] + len(b"<variable_types>")
    types = struct.unpack(f"<{len(codes)}H", raw[types_at: types_at + 2 * len(codes)])
    widths = [_TYPE_WIDTH[t] for t in types]
    data_at = offsets[9]
    if raw[data_at: data_at + 6] != b"<data>":
        raise ValueError("data section not where the map says")
    data_at += 6
    record = sum(widths)
    n = len(codes[0])
    if offsets[10] - len(b"</data>") - data_at != n * record:
        raise ValueError("data section is not n fixed-width records")
    column_at = np.concatenate(([0], np.cumsum(widths)[:-1]))
    for j, column in enumerate(codes):
        for i in np.flatnonzero(column != ""):
            start = data_at + i * record + int(column_at[j])
            raw[start: start + widths[j]] = _missing_bytes(types[j], column[i])
    path.write_bytes(bytes(raw))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--uppercase", action="store_true", help="write RAND's documented upper-case names")
    parser.add_argument("--households", type=int, default=N_HOUSEHOLDS)
    args = parser.parse_args()
    fixture = build_fixture(args.households)
    path = write_fixture(args.path, fixture, uppercase=args.uppercase)
    print(f"wrote {fixture.n} people x {len(fixture.columns)} columns to {path}")


if __name__ == "__main__":
    main()
