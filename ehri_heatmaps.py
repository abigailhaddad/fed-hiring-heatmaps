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

PLAN_FILTERS = {
    "gs":    "AND pay_plan_code = 'GS'",
    "gs+gg": "AND pay_plan_code IN ('GS', 'GG')",
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


@functools.lru_cache(maxsize=64)
def _fetch(series, pay_plans):
    con = _connection()
    lst = "[" + ",".join(f"'{u}'" for u in _monthly_urls()) + "]"
    src = f"read_parquet({lst})"
    conds = [f"personnel_action_effective_date_yyyymm >= '{START_MONTH}'"]
    if series is not None:
        conds.append(f"occupational_series_code = '{series}'")
    where = " AND ".join(conds)
    if series is None:
        name = "All occupations"
    else:
        row = con.execute(
            f"SELECT occupational_series FROM {src} WHERE {where} "
            f"GROUP BY 1 ORDER BY SUM(TRY_CAST(count AS BIGINT)) DESC LIMIT 1"
        ).fetchone()
        name = row[0] if row else series
    df = con.execute(f"""
        SELECT grade, education_level_code AS edu,
               SUM(TRY_CAST(count AS BIGINT)) AS hires
        FROM {src}
        WHERE {where} {PLAN_FILTERS[pay_plans]}
        GROUP BY 1, 2
    """).df()
    return df, name


def _matrix(df):
    df = df[df["grade"].isin(GS_GRADES)].copy()
    df["row"] = df["edu"].map(lambda c: CODE_TO_EDU.get(c, "≤ High school"))
    df["col"] = df["grade"].map(_grade_group)
    piv = (df.groupby(["row", "col"])["hires"].sum().reset_index()
             .pivot(index="row", columns="col", values="hires"))
    return piv.reindex(index=EDU_ORDER, columns=GRADE_ORDER).fillna(0)


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
