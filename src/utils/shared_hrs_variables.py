"""The extended HRS dictionary: every Yan (2026) risk factor RAND carries, as a recode table (D26).

Yan, "Revisiting the social determinants of health with explainable AI", AJE 195(3),
681-688, Table S2, lists 61 HRS risk factors in seven domains, taken from Puterman et
al.'s one-row-per-person file. Here each factor that the RAND Longitudinal File carries
is rebuilt from RAND itself, wave by wave, so that it can be a row of the life image;
factors fixed for life are static covariates instead. The factors RAND does not carry
(purpose, mastery, optimism, pessimism, hopelessness, constraints, anxiety, anger,
hostility, religiosity, social support and strain, discrimination, neighbourhood, life
events, food insecurity and stamps, most childhood items) need the raw core and
Leave-Behind files and are not here.

Each row has the fields of the user's SHARE / ELSA recode records (``varname``,
``conventional_name``, ``varname_in_raw``, ``domain``, ``available_waves``,
``recode_type``, ``reverse_code``, ``notes``), plus ``paper_factor`` -- the Table S2
name it recovers, if any -- and whether the paper had that factor in SHARE and ELSA.
Directions are natural (higher = more of the thing); the paper's "Lower X" reversals
are not applied, so ``reverse_code`` is False throughout.

The 12 core channels (``shared_hrs_ingest.CHANNELS``, the simulation's three streams)
are a subset and keep their own order; ``EXTENDED`` orders rows by domain.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from utils.shared_hrs_ingest import (
    ALL_WAVES,
    CHANNELS,
    LB_WAVES,
    N_WAVES,
    Channel,
    RandFrame,
    Source,
    _last_true,
    _pick,
    _wave_matrix,
)

CORE = {c.name: c for c in CHANNELS}


# --- recodes (codebook page in brackets) -------------------------------------------
def _codes_as_zero(stem: str, codes: tuple[str, ...]):
    """A recode where some missing codes mean 0, e.g. "not in the labour force" is not unemployed."""

    def recode(values: np.ndarray, wave: int, frame: RandFrame) -> np.ndarray:
        code = frame.codes[stem.format(w=wave)].to_numpy()
        return np.where(np.isin(code, codes), 0.0, values)

    recode.__name__ = f"codes_{'_'.join(c.strip('.') for c in codes)}_as_zero"
    return recode


def _not_homeowner(values: np.ndarray, wave: int, frame: RandFrame) -> np.ndarray:
    # HwAHOUS is the value of the primary residence [1203]; 0 means none is owned.
    return np.where(np.isnan(values), np.nan, (values <= 0).astype(float))


def _married_partnered(values: np.ndarray, wave: int, frame: RandFrame) -> np.ndarray:
    # RwMSTAT 1 married, 2 married spouse absent, 3 partnered [177].
    return np.where(np.isnan(values), np.nan, np.isin(values, (1, 2, 3)).astype(float))


def _separated_divorced(values: np.ndarray, wave: int, frame: RandFrame) -> np.ndarray:
    # RwMSTAT 4 separated, 5 divorced, 6 separated/divorced [177].
    return np.where(np.isnan(values), np.nan, np.isin(values, (4, 5, 6)).astype(float))


def _weekly_plus(values: np.ndarray, wave: int, frame: RandFrame) -> np.ndarray:
    # 1 every day, 2 more than once a week ... 5 never [586]; as vigorous activity (D20).
    return np.where(np.isnan(values), np.nan, (values <= 2).astype(float))


SLEEP_TROUBLE = ("r{w}sleepfal", "r{w}sleepwkn", "r{w}sleepwke")  # 1 most of the time .. 3 rarely [497]
SLEEP_RESTED = "r{w}sleeprt"  # 1 most of the time rested .. 3 rarely [498]


def _sleep_problems(values: np.ndarray, wave: int, frame: RandFrame) -> np.ndarray:
    """Mean of the four sleep items scored 0 (no problem) to 1 (most of the time); needs 2+ items."""
    items = [(3.0 - frame.values[s.format(w=wave)].to_numpy()) / 2.0 for s in SLEEP_TROUBLE]
    items.append((frame.values[SLEEP_RESTED.format(w=wave)].to_numpy() - 1.0) / 2.0)
    stacked = np.column_stack(items)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a person who answered none
        mean = np.nanmean(stacked, axis=1)
    return np.where(np.isfinite(stacked).sum(axis=1) >= 2, mean, np.nan)


# --- the table --------------------------------------------------------------------------
@dataclass(frozen=True)
class Variable:
    varname: str
    conventional_name: str
    domain: str
    sources: tuple[Source, ...]
    recode_type: str = "direct"
    reverse_code: bool = False
    paper_factor: str | None = None  # the Table S2 name it recovers
    paper_in_SHARE: bool = False
    paper_in_ELSA: bool = False
    notes: str = ""

    @property
    def varname_in_raw(self) -> str:
        return ", ".join(s.stem.upper().replace("{W}", "w") for s in self.sources)

    @property
    def available_waves(self) -> tuple[int, ...]:
        return tuple(sorted({w for s in self.sources for w in s.waves}))

    def channel(self) -> Channel:
        return Channel(self.varname, self.domain, self.sources)


def _core(name: str, conventional: str, domain: str, **kw) -> Variable:
    return Variable(name, conventional, domain, CORE[name].sources, **kw)


W = lambda first, last=N_WAVES: tuple(range(first, last + 1))  # noqa: E731

DOMAINS = ("socioeconomic", "adversity", "connections", "psychological", "behaviours")

EXTENDED_VARIABLES: tuple[Variable, ...] = (
    # socioeconomic
    _core("income", "Income", "socioeconomic", recode_type="cpi_real_asinh", paper_factor="Income", paper_in_ELSA=True),
    _core("wealth", "Wealth", "socioeconomic", recode_type="cpi_real_asinh", paper_factor="Wealth", paper_in_SHARE=True, paper_in_ELSA=True),
    _core("poverty", "Below poverty threshold", "socioeconomic"),
    _core("working", "Working for pay", "socioeconomic"),
    Variable("unemployed", "Unemployed", "socioeconomic",
             (Source("r{w}unemp", ALL_WAVES, _codes_as_zero("r{w}unemp", (".x", ".a"))),),
             recode_type="codes_x_a_as_zero", paper_factor="History of Unemployment", paper_in_SHARE=True, paper_in_ELSA=True,
             notes="RwUNEMP [1914]; .X not in labour force and .A 65+ presumed retired count as 0, not missing"),
    Variable("medicaid", "Covered by Medicaid", "socioeconomic", (Source("r{w}govmd", ALL_WAVES),),
             paper_factor="History of Medicaid", notes="RwGOVMD [1615]; the paper's is 'ever', this is per wave"),
    Variable("not_homeowner", "Does not own home", "socioeconomic", (Source("h{w}ahous", ALL_WAVES, _not_homeowner),),
             recode_type="zero_value_is_1", paper_factor="History of Renting", paper_in_SHARE=True, paper_in_ELSA=True,
             notes="HwAHOUS = 0 [1203]: renting or otherwise not owning; the household's, as income"),
    Variable("financial_strain", "Ongoing financial strain", "socioeconomic", (Source("r{w}lbfinprb", (8,) + W(10)),),
             paper_factor="Recent Financial Difficulties",
             notes="RwLBFINPRB [2081], 1 did not happen .. 4 very upsetting; Leave-Behind waves 8, 10-16 (not asked in 9, "
                   "like chronic stress; found by the schema report 2026-10-25)"),
    # adversity
    _core("chronic_stress", "Ongoing chronic stress", "adversity",
          notes="nearest RAND measure to the paper's adulthood psychosocial adversity; not the same scale"),
    # connections
    Variable("married_partnered", "Married or partnered", "connections", (Source("r{w}mstat", ALL_WAVES, _married_partnered),),
             recode_type="mstat_1_2_3_is_1",
             notes="RwMSTAT [177] currently married, spouse absent or partnered; 0 = separated, divorced, widowed "
                   "or never married. Not a paper factor (added at the user's request, D27)"),
    Variable("separated_divorced", "Separated or divorced", "connections", (Source("r{w}mstat", ALL_WAVES, _separated_divorced),),
             recode_type="mstat_4_5_6_is_1", paper_factor="History of Divorce", paper_in_SHARE=True, paper_in_ELSA=True,
             notes="RwMSTAT [177] currently separated/divorced; times_divorced carries the history"),
    Variable("times_divorced", "Number of divorces", "connections", (Source("r{w}mdiv", ALL_WAVES),),
             paper_factor="History of Divorce", paper_in_SHARE=True, paper_in_ELSA=True, notes="RwMDIV [201], 0-6"),
    # psychological (Leave-Behind, from 2006, each person every other wave)
    _core("loneliness", "Loneliness (3 items)", "psychological", paper_factor="Loneliness", paper_in_ELSA=True),
    _core("life_satisfaction", "Life satisfaction", "psychological", recode_type="wave8_rescaled",
          paper_factor="Lower Life Satisfaction", paper_in_ELSA=True),
    _core("negative_affect", "Negative affect", "psychological", paper_factor="Negative Affectivity", paper_in_SHARE=True),
    Variable("positive_affect", "Positive affect", "psychological", (Source("r{w}lbposaffect", W(9)),),
             paper_factor="Lower Positive Affectivity", paper_in_SHARE=True, notes="RwLBPOSAFFECT [2095], 13 items, 1-5; wave 9 on"),
    Variable("neuroticism", "Neuroticism", "psychological", (Source("r{w}lbneur", LB_WAVES),),
             paper_factor="Lower Neuroticism", notes="RwLBNEUR [2070], 1-4"),
    Variable("extroversion", "Extroversion", "psychological", (Source("r{w}lbext", LB_WAVES),),
             paper_factor="Lower Extroversion", notes="RwLBEXT [2070], 1-4"),
    Variable("openness", "Openness to experience", "psychological", (Source("r{w}lbopen", LB_WAVES),),
             paper_factor="Lower Openness to Experiences", notes="RwLBOPEN [2070], 1-4"),
    Variable("agreeableness", "Agreeableness", "psychological", (Source("r{w}lbagr", LB_WAVES),),
             paper_factor="Lower Agreeableness", notes="RwLBAGR [2070], 1-4"),
    Variable("conscientiousness", "Conscientiousness (5 items)", "psychological", (Source("r{w}lbcon5", LB_WAVES),),
             paper_factor="Lower Conscientiousness", notes="RwLBCON5 [2071], 1-4; the 5-item version spans waves 8-16"),
    Variable("depressive_symptoms", "CES-D score", "psychological", (Source("r{w}cesd", W(2)),),
             notes="RwCESD [335], 0-8; not a paper factor, the core's only every-wave psychological measure"),
    # behaviours
    _core("bmi", "Body mass index", "behaviours"),
    _core("smoking", "Current smoker", "behaviours", paper_factor="Current Smoker", paper_in_SHARE=True, paper_in_ELSA=True),
    Variable("ever_smoked", "Ever smoked", "behaviours", (Source("r{w}smokev", ALL_WAVES),),
             paper_factor="History of Smoking", paper_in_SHARE=True, paper_in_ELSA=True, notes="RwSMOKEV [623]"),
    _core("drinking_days", "Days a week drinking", "behaviours", paper_factor="Alcohol Abuse", paper_in_ELSA=True,
          notes="frequency, not the paper's CAGE abuse screen, which RAND lacks"),
    Variable("drinks_per_day", "Drinks per drinking day", "behaviours", (Source("r{w}drinkn", W(3)),),
             paper_factor="Alcohol Abuse", paper_in_ELSA=True, notes="RwDRINKN [600]"),
    _core("vigorous_activity", "Vigorous activity weekly+", "behaviours", recode_type="vigact_or_vgactx_le_2",
          paper_factor="Low/No Vigorous Activity", paper_in_SHARE=True, paper_in_ELSA=True),
    Variable("moderate_activity", "Moderate activity weekly+", "behaviours", (Source("r{w}mdactx", W(7), _weekly_plus),),
             recode_type="le_2_is_1", paper_factor="Low/No Moderate Activity", paper_in_SHARE=True,
             notes="RwMDACTX [586]; 1 if more than once a week; waves 7-16 only"),
    Variable("sleep_problems", "Sleep problems", "behaviours",
             (Source(SLEEP_TROUBLE[0], (6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), _sleep_problems,
                     extra=SLEEP_TROUBLE[1:] + (SLEEP_RESTED,)),),
             recode_type="mean_of_4_items_0_1", paper_factor="Sleep Problems", paper_in_SHARE=True,
             notes="RwSLEEPFAL/WKN/WKE/RT [497-498]: trouble items 1 most..3 rarely -> (3-x)/2, rested (x-1)/2; "
                   "mean of 2+ answered; wave 9 and 11 mostly 'answered previous wave' (missing)"),
)
EXTENDED: tuple[Channel, ...] = tuple(v.channel() for v in EXTENDED_VARIABLES)
CHANNEL_SETS = {"core": CHANNELS, "extended": EXTENDED}


def channel_domains(channels: tuple[Channel, ...]) -> list[str]:
    """The group each row belongs to: the stream (core) or the domain (extended)."""
    return [c.stream for c in channels]


# --- static covariates ----------------------------------------------------------------------
@dataclass(frozen=True)
class Static:
    varname: str
    conventional_name: str
    domain: str
    varname_in_raw: str
    paper_factor: str | None
    notes: str = ""


STATICS: tuple[Static, ...] = (
    Static("birth_year", "Birth year", "demographic", "RABYEAR", None, "the cohort covariate (D21)"),
    Static("female", "Female", "demographic", "RAGENDER", "Male", "RAGENDER = 2"),
    Static("black", "Black", "demographic", "RARACEM", "Black", "RARACEM = 2 [144]"),
    Static("other_race", "Other race", "demographic", "RARACEM", None, "RARACEM = 3"),
    Static("hispanic", "Hispanic", "demographic", "RAHISPAN", "Hispanic", "RAHISPAN = 1 [147]"),
    Static("foreign_born", "Foreign born", "demographic", "RABPLACE", "Foreign Born",
           "RABPLACE 11 or 13, not US [230]; 12 US territory counts as US"),
    Static("education_years", "Years of education", "socioeconomic", "RAEDYRS", "Lower Education", "0-17 [162]"),
    Static("father_education", "Father's years of education", "childhood", "RAFEDUC", "Father Education",
           "[171]; 7.5 / 8.5 are 'less than / more than 8'; .D frequent, so a missing indicator goes with it"),
    Static("mother_education", "Mother's years of education", "childhood", "RAMEDUC", "Mother Education", "[171]"),
    Static("never_married", "Never married", "connections", "RwMNEV", "Never Married",
           "RwMNEV at the first interview in the image [200]; .N 'no marriages, likely never married' counts as 1"),
    Static("lower_occupation", "Manual or service longest job", "socioeconomic", "RwJLOCC", "Lower Occupational Status",
           "RwJLOCC 1980 Census [2010]: 5-16 -> 1, 1-4 -> 0, armed forces and other coding (.B, later cohorts) missing"),
)
STATIC_SETS = {"core": ("birth_year", "female"), "extended": tuple(s.varname for s in STATICS)}
STATIC_COLUMNS = ("rabyear", "ragender", "raracem", "rahispan", "rabplace", "raedyrs", "rafeduc", "rameduc")
STATIC_WAVE_STEMS = ("r{w}mnev", "r{w}jlocc")


def static_columns() -> list[str]:
    return list(STATIC_COLUMNS) + [s.format(w=w) for s in STATIC_WAVE_STEMS for w in ALL_WAVES]


def static_matrix(frame: RandFrame, names: tuple[str, ...], first_wave: np.ndarray, last_wave: np.ndarray) -> np.ndarray:
    """``(n, len(names))`` raw static covariates, NaN where unknown.

    ``first_wave`` / ``last_wave`` (0-based, -1 none) are the person's first interview in
    the image and the status interview: never-married is read at the first, the longest
    occupation from the last valid report up to the second.
    """
    v = frame.values
    columns = []
    for name in names:
        if name == "birth_year":
            col = v["rabyear"].to_numpy()
        elif name == "female":
            col = _indicator(v["ragender"].to_numpy(), (2,))
        elif name == "black":
            col = _indicator(v["raracem"].to_numpy(), (2,))
        elif name == "other_race":
            col = _indicator(v["raracem"].to_numpy(), (3,))
        elif name == "hispanic":
            col = _indicator(v["rahispan"].to_numpy(), (1,))
        elif name == "foreign_born":
            col = _indicator(v["rabplace"].to_numpy(), (11, 13))
        elif name in ("education_years", "father_education", "mother_education"):
            col = v[{"education_years": "raedyrs", "father_education": "rafeduc", "mother_education": "rameduc"}[name]].to_numpy()
        elif name == "never_married":
            mnev = _wave_matrix(frame, "r{w}mnev")
            codes = np.column_stack([frame.codes[f"r{w}mnev"].to_numpy() for w in ALL_WAVES])
            mnev = np.where(codes == ".n", 1.0, mnev)
            col = _pick(mnev, first_wave)
        elif name == "lower_occupation":
            occ = _wave_matrix(frame, "r{w}jlocc")
            valid = (occ >= 1) & (occ <= 16) & (np.arange(N_WAVES)[None, :] <= last_wave[:, None])
            col = _pick(np.where(occ >= 5, 1.0, 0.0), _last_true(valid))
        else:
            raise ValueError(f"unknown static covariate {name!r}")
        columns.append(np.asarray(col, dtype=float))
    return np.column_stack(columns) if columns else np.zeros((frame.n, 0))


def _indicator(values: np.ndarray, ones: tuple[int, ...]) -> np.ndarray:
    return np.where(np.isnan(values), np.nan, np.isin(values, ones).astype(float))

