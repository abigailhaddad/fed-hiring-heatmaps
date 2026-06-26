"""
Grade x education heatmaps of new federal hires, for any occupational series.

One public entry point, one clean (Substack-ready) layout:

    from ehri_heatmaps import accession_heatmap
    accession_heatmap("2210")          # IT Management
    accession_heatmap()                # all occupations
    accession_heatmap("1530", pay_plans="gs")   # GS only (default is GS+GG)

The grid is collapsed to **5 education buckets** x **4 grade groups** so it reads
at a glance, with percentage totals in the margins and the degree->grade
"could-qualify" staircase outlined.

Data: the public HuggingFace dataset `impactproject/opm-ehri-data` (OPM/EHRI
federal "accessions" = new hires). Parquet is streamed remotely with DuckDB —
nothing is downloaded to disk. Each source row is pre-aggregated, so counts are
SUM(count), not row counts. Scope per call: one occupational_series_code (or all),
accession effective month >= START_MONTH.
"""
import re
import calendar
import functools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import duckdb
from huggingface_hub import HfApi

REPO = "impactproject/opm-ehri-data"
START_MONTH = "202101"          # earliest accession month included
CMAP = "Oranges"                # single-hue ramp (near-white -> dark orange)

# --- education: education_level_code -> one of 5 buckets (ascending) --------
SIMPLE_EDU = [
    ("≤ High school",            {"01", "02", "03", "04", "05", "06", "*", None}),
    ("Some college / Associate", {"07", "08", "09", "10", "11", "12"}),
    ("Bachelor's / post-bach.",  {"13", "14"}),
    ("Master's / professional",  {"15", "16", "17", "18", "19", "20"}),  # prof. credited at GS-9
    ("Doctorate",                {"21", "22"}),
]
CODE_TO_EDU = {c: label for label, codes in SIMPLE_EDU for c in codes}
EDU_ORDER = [label for label, _ in SIMPLE_EDU]

# --- grade: GS/GG grade -> one of 4 groups ----------------------------------
GS_GRADES = {f"{g:02d}" for g in range(1, 16)}   # GS-01 .. GS-15
GRADE_ORDER = ["≤ GS-7", "GS-8–9", "GS-10–11", "GS-12+"]


def _grade_group(g):
    n = int(g)
    if n <= 7:
        return "≤ GS-7"
    if n <= 9:
        return "GS-8–9"
    if n <= 11:
        return "GS-10–11"
    return "GS-12+"


# --- qualifying staircase: education rows a degree could qualify for, by group
# (OPM group-coverage standard: bachelor's->GS-7, master's->GS-9, Ph.D.->GS-11;
# a higher degree also covers the lower groups.)
QUAL = {
    "≤ GS-7":   {"Bachelor's / post-bach.", "Master's / professional", "Doctorate"},
    "GS-8–9":   {"Master's / professional", "Doctorate"},
    "GS-10–11": {"Doctorate"},
}

PLAN_FILTERS = {                         # bare predicates on pay_plan_code
    "gs":    "pay_plan_code = 'GS'",
    "gs+gg": "pay_plan_code IN ('GS', 'GG')",
}


# --------------------------------------------------------------------------- #
# data access (cached so repeated calls in a notebook are cheap)
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=1)
def _connection():
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    return con


@functools.lru_cache(maxsize=1)
def _monthly_urls():
    """Latest version per month from START_MONTH onward, as hf:// URLs."""
    files = [f for f in HfApi().list_repo_files(REPO, repo_type="dataset")
             if f.startswith("accessions/")]
    best = {}
    for f in files:
        m = re.search(r"accessions_(\d{6})_v(\d+)\.parquet", f)
        if not m:
            continue
        month, ver = m.group(1), int(m.group(2))
        if month < START_MONTH:
            continue
        if month not in best or ver > best[month][0]:
            best[month] = (ver, f)
    return tuple(f"hf://datasets/{REPO}/{v[1]}" for _, v in sorted(best.items()))


def _month_label(yyyymm):
    return f"{calendar.month_name[int(yyyymm[4:6])]} {yyyymm[:4]}"


@functools.lru_cache(maxsize=1)
def _date_range():
    """e.g. 'January 2021 – April 2026', from the actual months present."""
    months = sorted(re.search(r"accessions_(\d{6})_", u).group(1) for u in _monthly_urls())
    return f"{_month_label(months[0])} – {_month_label(months[-1])}"


def _counts_sql(src, series, pay_plans):
    """The DuckDB query that pulls the per-(grade, education) hire counts."""
    conds = [f"personnel_action_effective_date_yyyymm >= '{START_MONTH}'"]
    if series is not None:
        conds.append(f"occupational_series_code = '{series}'")
    conds.append(PLAN_FILTERS[pay_plans])
    where = "\n      AND ".join(conds)
    return (
        "SELECT grade,\n"
        "       education_level_code AS edu,\n"
        "       SUM(TRY_CAST(count AS BIGINT)) AS hires   -- rows are pre-aggregated\n"
        f"FROM {src}\n"
        f"WHERE {where}\n"
        "GROUP BY grade, education_level_code"
    )


def build_sql(series=None, pay_plans="gs+gg"):
    """The query `accession_heatmap` runs, in readable form (for transparency).

    The real `FROM` is `read_parquet([...])` over every monthly accession file;
    here it's shown as a placeholder so the query fits on screen.
    """
    pay_plans = pay_plans.lower().replace("gsgg", "gs+gg")
    n = len(_monthly_urls())
    src = (f"read_parquet([ {n} monthly accession files,\n"
           f"               'hf://datasets/{REPO}/accessions/accessions_YYYYMM_v*.parquet' ])")
    return _counts_sql(src, None if series is None else str(series), pay_plans)


@functools.lru_cache(maxsize=64)
def _fetch(series, pay_plans):
    con = _connection()
    src = "read_parquet([" + ",".join(f"'{u}'" for u in _monthly_urls()) + "])"
    if series is None:
        name = "All occupations"
    else:
        row = con.execute(
            f"SELECT occupational_series FROM {src} "
            f"WHERE personnel_action_effective_date_yyyymm >= '{START_MONTH}' "
            f"AND occupational_series_code = '{series}' "
            f"GROUP BY 1 ORDER BY SUM(TRY_CAST(count AS BIGINT)) DESC LIMIT 1"
        ).fetchone()
        name = row[0] if row else series
    df = con.execute(_counts_sql(src, series, pay_plans)).df()
    return df, name


def _matrix(df):
    df = df[df["grade"].isin(GS_GRADES)].copy()
    df["row"] = df["edu"].map(lambda c: CODE_TO_EDU.get(c, "≤ High school"))
    df["col"] = df["grade"].map(_grade_group)
    piv = (df.groupby(["row", "col"])["hires"].sum().reset_index()
             .pivot(index="row", columns="col", values="hires"))
    return piv.reindex(index=EDU_ORDER, columns=GRADE_ORDER).fillna(0)


# --------------------------------------------------------------------------- #
# transparency helpers: see the data behind a figure, step by step
# --------------------------------------------------------------------------- #
def dataset_info():
    """What's being read: dataset, number of monthly files, and date range."""
    return {"dataset": REPO,
            "monthly_accession_files": len(_monthly_urls()),
            "date_range": _date_range()}


def fetch_counts(series=None, pay_plans="gs+gg"):
    """Raw query result: one row per (grade, education_level_code) with hires.

    This is exactly what `build_sql(series, pay_plans)` returns, run against the
    monthly parquet files. Education codes and grades are still raw here.
    """
    pay_plans = pay_plans.lower().replace("gsgg", "gs+gg")
    df, _ = _fetch(None if series is None else str(series), pay_plans)
    return df.sort_values("hires", ascending=False).reset_index(drop=True)


def crosstab(series=None, pay_plans="gs+gg"):
    """The 5 x 4 table of hires the heatmap draws (after bucketing)."""
    pay_plans = pay_plans.lower().replace("gsgg", "gs+gg")
    df, _ = _fetch(None if series is None else str(series), pay_plans)
    return _matrix(df).astype(int)


def bucket_maps():
    """Two small tables documenting how raw codes/grades collapse into buckets."""
    edu = pd.DataFrame(
        [(label, ", ".join(sorted(c for c in codes if c))) for label, codes in SIMPLE_EDU],
        columns=["education bucket", "education_level_code(s)"])
    grade = pd.DataFrame(
        [(f"GS-{g}", _grade_group(g)) for g in sorted(GS_GRADES)],
        columns=["grade", "grade group"])
    return edu, grade


# --------------------------------------------------------------------------- #
# occupation-aware: join hiring to OPM education-requirement tiers
#
# Were these hires' degrees what qualified them for their grade — or did they
# come in on experience? We classify each hire by the qualifying staircase
# (degree could open that grade) and join the occupational series to OPM's
# published education-requirement tier (from the opm-educ-req project).
# --------------------------------------------------------------------------- #
OPM_TIERS_URL = "https://raw.githubusercontent.com/abigailhaddad/opm-educ-req/main/opm_series_tiers.json"

# (tier, mandatory_type) -> readable category, matching the opm-educ-req site
TIER_CATEGORY = {
    ("mandatory", "professional"):  "Mandatory: Professional",
    ("mandatory", "qualification"): "Mandatory: Qualification",
    ("optional", ""):               "Optional",
    ("none", ""):                   "None",
}
TIER_ORDER = ["Mandatory: Professional", "Mandatory: Qualification",
              "Optional", "None", "Unclassified"]

# the (grade group, education bucket) cells where a degree could qualify the hire
_QUAL_PAIRS = {(g, e) for g, es in QUAL.items() for e in es}


@functools.lru_cache(maxsize=1)
def opm_tiers():
    """series_num -> OPM education-requirement tier (from opm-educ-req)."""
    df = pd.read_json(OPM_TIERS_URL)
    df["series_num"] = df["series_num"].astype(str).str.zfill(4)
    df["mandatory_type"] = df["mandatory_type"].fillna("").astype(str)
    df["category"] = [TIER_CATEGORY.get((t, m), "Optional")
                      for t, m in zip(df["tier"], df["mandatory_type"])]
    return df[["series_num", "series_title", "tier", "mandatory_type", "category"]]


@functools.lru_cache(maxsize=4)
def _series_counts(pay_plans):
    """Per (series, grade, education) hire counts across all occupations."""
    con = _connection()
    src = "read_parquet([" + ",".join(f"'{u}'" for u in _monthly_urls()) + "])"
    return con.execute(f"""
        SELECT occupational_series_code AS series, grade,
               education_level_code AS edu,
               SUM(TRY_CAST(count AS BIGINT)) AS hires
        FROM {src}
        WHERE personnel_action_effective_date_yyyymm >= '{START_MONTH}'
          AND {PLAN_FILTERS[pay_plans]}
        GROUP BY 1, 2, 3
    """).df()


def qualifying_by_series(pay_plans="gs+gg"):
    """Per occupational series: hires, how many a degree could have qualified,
    the share, and the OPM education-requirement category."""
    pay_plans = pay_plans.lower().replace("gsgg", "gs+gg")
    df = _series_counts(pay_plans)
    df = df[df["grade"].isin(GS_GRADES)].copy()
    df["series"] = df["series"].astype(str).str.zfill(4)
    df["edu_bucket"] = df["edu"].map(lambda c: CODE_TO_EDU.get(c, "≤ High school"))
    df["grade_group"] = df["grade"].map(_grade_group)
    df["qual_hires"] = [h if (g, e) in _QUAL_PAIRS else 0
                        for h, g, e in zip(df["hires"], df["grade_group"], df["edu_bucket"])]
    g = (df.groupby("series", as_index=False)
           .agg(hires=("hires", "sum"), degree_qualifying=("qual_hires", "sum")))
    g["pct_degree_qualifying"] = (100 * g["degree_qualifying"] / g["hires"]).round(1)
    g = g.merge(opm_tiers()[["series_num", "series_title", "category"]],
                left_on="series", right_on="series_num", how="left").drop(columns="series_num")
    g["category"] = g["category"].fillna("Unclassified")
    return g.sort_values("hires", ascending=False).reset_index(drop=True)


def qualifying_by_tier(pay_plans="gs+gg"):
    """Roll up to OPM education-requirement category: hires and the share whose
    degree could have qualified them for their grade."""
    g = qualifying_by_series(pay_plans)
    agg = (g.groupby("category", as_index=False)
             .agg(hires=("hires", "sum"), degree_qualifying=("degree_qualifying", "sum")))
    agg["pct_degree_qualifying"] = (100 * agg["degree_qualifying"] / agg["hires"]).round(1)
    agg["__o"] = agg["category"].map(lambda c: TIER_ORDER.index(c) if c in TIER_ORDER else 99)
    return agg.sort_values("__o").drop(columns="__o").reset_index(drop=True)


def qualifying_overall(pay_plans="gs+gg"):
    """Headline: across all hires, the share whose degree could have qualified
    them for their grade (the rest came in on experience)."""
    g = qualifying_by_series(pay_plans)
    h, q = int(g["hires"].sum()), int(g["degree_qualifying"].sum())
    return {"hires": h, "degree_could_qualify": q,
            "pct_degree_could_qualify": round(100 * q / h, 1),
            "pct_experience_based": round(100 * (h - q) / h, 1)}


def qualifying_tier_chart(pay_plans="gs+gg", save=False, out=None):
    """Bar chart: share of new hires whose degree could qualify them for their
    grade, by OPM education-requirement category, with the all-hires average."""
    tier = qualifying_by_tier(pay_plans)
    tier = tier[tier["category"] != "Unclassified"]
    ov = qualifying_overall(pay_plans)
    cats = list(tier["category"])[::-1]          # plot top-to-bottom in TIER_ORDER
    pct = list(tier["pct_degree_qualifying"])[::-1]
    hires = list(tier["hires"].astype(int))[::-1]
    qual = list(tier["degree_qualifying"].astype(int))[::-1]

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(10, 3.6))
    y = range(len(cats))
    ax.barh(list(y), pct, color="#d4711f", height=0.62, zorder=3)
    ax.axvline(ov["pct_degree_could_qualify"], color="#1565c0", lw=1.8, ls="--", zorder=4)
    ax.text(ov["pct_degree_could_qualify"], -0.95,
            f"all-hires average: {ov['pct_degree_could_qualify']:.0f}%", color="#1565c0",
            fontsize=9, fontweight="bold", va="center", ha="center")

    for i, (p, h, qn) in enumerate(zip(pct, hires, qual)):
        ax.text(p + 0.4, i, f"{p:.0f}%   ({qn:,} of {h:,})",
                va="center", ha="left", fontsize=9, color="#333333")

    ax.set_yticks(list(y))
    ax.set_yticklabels(cats, fontsize=10)
    ax.set_xlim(0, max(pct) * 1.45)
    ax.set_ylim(-1.3, len(cats) - 0.4)
    ax.set_xlabel("Share of new hires whose degree could qualify them for their grade", fontsize=9.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#eee")
    ax.set_title("Most federal hires aren't qualified into their grade by a degree",
                 fontsize=14, fontweight="bold", loc="left", pad=24)
    ax.annotate(
        f"GS + GG accessions, {_date_range()}   ·   n = {ov['hires']:,}   ·   "
        "OPM/EHRI hires × OPM education-requirement tier (opm-educ-req)",
        xy=(0, 1), xycoords="axes fraction", xytext=(0, 8),
        textcoords="offset points", fontsize=9, color="#666666", ha="left")
    fig.tight_layout()
    if save or out:
        path = out or "qualifying_by_tier.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        print(f"wrote {path}")
    return fig


# --------------------------------------------------------------------------- #
# the one public function
# --------------------------------------------------------------------------- #
def accession_heatmap(series=None, pay_plans="gs+gg", highlight_quals=True,
                      totals=True, save=False, out=None):
    """Render the grade x education heatmap of new federal hires.

    Parameters
    ----------
    series : str or None
        Occupational series code, e.g. "2210" (IT Mgmt), "1550" (Comp Sci),
        "1560" (Data Sci), "1530" (Statistics). None -> all occupations.
    pay_plans : {"gs+gg", "gs"}
        "gs+gg" (default) = General Schedule plus the GG plan (same grade scale);
        "gs" = General Schedule only.
    highlight_quals : bool
        Outline cells where a hire's degree could qualify them for that grade
        (bachelor's->GS-7, master's->GS-9, Ph.D.->GS-11; a higher degree also
        covers the lower groups) and show the headline share box. Default True.
    totals : bool   show the grey row/column percentage strips (default True).
    save : bool     write a PNG (default heatmap_<series|all>.png).
    out : str       explicit output path (implies save).

    Returns
    -------
    matplotlib.figure.Figure  (displays inline in a notebook)
    """
    pay_plans = pay_plans.lower().replace("gsgg", "gs+gg")
    if pay_plans not in PLAN_FILTERS:
        raise ValueError(f"pay_plans must be one of {list(PLAN_FILTERS)}")
    series = None if series is None else str(series)

    df, series_name = _fetch(series, pay_plans)
    mat = _matrix(df)
    fig = _plot(mat, series, series_name, pay_plans,
                show_totals=totals, highlight_quals=highlight_quals)
    if save or out:
        tag = series if series is not None else "all"
        path = out or f"heatmap_{tag}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        print(f"wrote {path}")
    return fig


def _plot(mat, series, series_name, pay_plans, show_totals=True, highlight_quals=True):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#cccccc",
        "figure.facecolor": "white",
    })
    data = mat.values
    nrows, ncols = data.shape
    grand = data.sum() or 1
    col_tot, row_tot = data.sum(axis=0), data.sum(axis=1)
    vmax = data.max() or 1

    fig, ax = plt.subplots(figsize=(2.6 + 1.7 * ncols, 1.9 + 0.74 * nrows))
    im = ax.imshow(data, aspect="auto", cmap=CMAP, vmin=0, vmax=vmax)

    # crisp white gridlines, no spines/ticks
    ax.set_xticks(np.arange(-.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, nrows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(range(ncols))
    ax.set_xticklabels(list(mat.columns), fontsize=11)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(list(mat.index), fontsize=11)

    def fmt(v):
        return f"{int(v):,}" if v else ""

    def pct(v):
        return f"{round(100 * v / grand)}%" if v else ""

    qual_color = "#1565c0"
    active = QUAL if highlight_quals else {}
    qual_cells = {(i, j)
                  for j, c in enumerate(mat.columns)
                  for i, b in enumerate(mat.index)
                  if c in active and b in active[c] and data[i, j] > 0}

    for i in range(nrows):
        for j in range(ncols):
            v = data[i, j]
            if not v:
                continue
            hot = (i, j) in qual_cells
            ax.text(j, i, fmt(v), ha="center", va="center",
                    fontsize=11.5 if hot else 11,
                    fontweight="bold" if hot else "normal",
                    color="white" if v > 0.55 * vmax else "#333333")
    for (i, j) in qual_cells:
        ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False,
                               edgecolor=qual_color, lw=2.5, zorder=5))

    # headline share box, centered below the chart
    if highlight_quals and qual_cells:
        qual_sum = sum(data[i, j] for (i, j) in qual_cells)
        ax.annotate(
            f"{qual_sum / grand:.0%} of hires (n = {int(qual_sum):,}) sit in an outlined "
            "cell — their degree could qualify them for that grade",
            xy=(0.5, 0), xycoords="axes fraction", xytext=(0, -48),
            textcoords="offset points", ha="center", va="top",
            fontsize=10, color="#0d3b66",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor=qual_color, linewidth=1.8))

    # margin totals: percentages only
    if show_totals:
        grey, dark = "#e8e8e8", "#bdbdbd"
        for i in range(nrows):
            ax.add_patch(Rectangle((ncols - .5, i - .5), 1, 1, facecolor=grey,
                                   edgecolor="white", lw=2, clip_on=False))
            if row_tot[i]:
                ax.text(ncols, i, pct(row_tot[i]), ha="center", va="center",
                        fontsize=10, fontweight="bold", color="#222222")
        for j in range(ncols):
            ax.add_patch(Rectangle((j - .5, nrows - .5), 1, 1, facecolor=grey,
                                   edgecolor="white", lw=2, clip_on=False))
            if col_tot[j]:
                ax.text(j, nrows, pct(col_tot[j]), ha="center", va="center",
                        fontsize=10, fontweight="bold", color="#222222")
        ax.add_patch(Rectangle((ncols - .5, nrows - .5), 1, 1, facecolor=dark,
                               edgecolor="white", lw=2, clip_on=False))
        ax.text(ncols, nrows, "100%", ha="center", va="center",
                fontsize=9, fontweight="bold", color="black")
        ax.text(-0.85, nrows, "%", ha="right", va="center",
                fontsize=9, fontweight="bold", color="#555555")

    pad = .5 if show_totals else -.5
    ax.set_xlim(-.5, ncols + pad)
    ax.set_ylim(nrows + pad, -.5)
    ax.set_xlabel("Grade group  (GS + GG)" if pay_plans == "gs+gg"
                  else "Grade group  (GS)", fontsize=10, labelpad=8)
    ax.set_ylabel("Education level", fontsize=10)

    title_name = series_name.title() if series_name.isupper() else series_name
    title = "All occupations" if series is None else f"{series}  {title_name}"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=42, loc="left")
    plans_label = "GS + GG" if pay_plans == "gs+gg" else "GS"
    ax.annotate(
        f"New {plans_label} hires by grade & education   ·   {_date_range()}   "
        f"·   n = {int(grand):,}   ·   OPM / EHRI",
        xy=(0, 1), xycoords="axes fraction", xytext=(0, 24),
        textcoords="offset points", fontsize=9.5, color="#666666", ha="left")
    if highlight_quals:
        ax.annotate(
            "Outlined: a hire's degree could qualify them for that grade "
            "(bachelor's→GS-7, master's→GS-9, Ph.D.→GS-11)",
            xy=(0, 1), xycoords="axes fraction", xytext=(0, 8),
            textcoords="offset points", fontsize=8.5, color="#1565c0", ha="left")

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    # usage: python ehri_heatmaps.py [SERIES|overall] [gs|gs+gg]
    import sys
    series, pay_plans = "2210", "gs+gg"
    for a in sys.argv[1:]:
        al = a.lower()
        if al in ("gs", "gsgg", "gs+gg"):
            pay_plans = "gs+gg" if al in ("gsgg", "gs+gg") else "gs"
        elif al in ("overall", "all", "none"):
            series = None
        else:
            series = a
    accession_heatmap(series, pay_plans=pay_plans, save=True)
