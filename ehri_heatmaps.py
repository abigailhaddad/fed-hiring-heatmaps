"""
Grade x education heatmaps of new federal hires for any occupational series.

One public entry point:

    from ehri_heatmaps import accession_heatmap
    accession_heatmap("2210")                 # GS hires only
    accession_heatmap("1530", all_plans=True) # GS+GG on the grade scale, rest pooled

Data: the public HuggingFace dataset `impactproject/opm-ehri-data` (OPM/EHRI
federal "accessions" = new hires). Parquet is streamed remotely with DuckDB —
nothing is downloaded to disk. Each source row is pre-aggregated, so counts are
SUM(count), not row counts.

Scope per call: occupational_series_code == <series>, accession effective month
>= START_MONTH (default Jan 2021).
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
OTHER_COL = "Other plans"       # all-plans mode: pool of non-GS/GG hires
CMAP = "Oranges"                # single-hue ramp (near-white -> dark orange)

# education_level_code -> bucket, in ascending order of attainment (row order)
EDU_BUCKETS = [
    ("Below HS / unknown", ["01", "02", "03", "05", "06", "*", None]),
    ("High school",        ["04"]),
    ("Some college",       ["07", "08", "09", "11", "12"]),
    ("Associate",          ["10"]),
    ("Bachelor's",         ["13"]),
    ("Post-bachelor's",    ["14"]),
    ("Master's",           ["17", "18"]),
    ("Doctorate",          ["21", "22"]),
    ("Other prof/adv",     ["15", "16", "19", "20"]),
]
CODE_TO_BUCKET = {c: name for name, codes in EDU_BUCKETS for c in codes}
EDU_ORDER = [name for name, _ in EDU_BUCKETS]
GRADE_ORDER = [f"{g:02d}" for g in range(1, 16)]  # GS-01 .. GS-15

# Cells where the degree alone can qualify a hire for that grade, under OPM's
# group-coverage qualification standard for professional/2-grade-interval work:
#   GS-5  <- bachelor's
#   GS-7  <- bachelor's (superior academic achievement) / 1 yr graduate
#   GS-9  <- master's / 2 yrs graduate
#   GS-11 <- Ph.D. / 3 yrs graduate
# A degree opens its ceiling grade AND every degree-entry grade below it, so the
# highlighted region is a staircase (e.g. a master's also covers GS-5 and GS-7).
QUAL_ENTRY_GRADES = ["05", "07", "09", "11"]
EDU_CEILING = {            # highest degree-entry grade each education bucket opens
    "Bachelor's":      7,
    "Post-bachelor's": 7,
    "Master's":        9,
    "Other prof/adv":  9,
    "Doctorate":       11,
}
QUAL_RULES = {
    g: {b for b, ceil in EDU_CEILING.items() if ceil >= int(g)}
    for g in QUAL_ENTRY_GRADES
}

# --------------------------------------------------------------------------- #
# simplified (Substack-friendly) groupings: 5 education rows x 4 grade columns
# --------------------------------------------------------------------------- #
SIMPLE_EDU_MAP = {            # 9 detailed buckets -> 5 readable rows
    "Below HS / unknown": "≤ High school",
    "High school":        "≤ High school",
    "Some college":       "Some college / Associate",
    "Associate":          "Some college / Associate",
    "Bachelor's":         "Bachelor's / post-bach.",
    "Post-bachelor's":    "Bachelor's / post-bach.",
    "Master's":           "Master's / professional",
    "Other prof/adv":     "Master's / professional",  # credited at GS-9 too
    "Doctorate":          "Doctorate",
}
SIMPLE_EDU_ORDER = ["≤ High school", "Some college / Associate",
                    "Bachelor's / post-bach.", "Master's / professional", "Doctorate"]
SIMPLE_GRADE_ORDER = ["≤ GS-7", "GS-8–9", "GS-10–11", "GS-12+"]


def _grade_group(g):
    n = int(g)
    if n <= 7:
        return "≤ GS-7"
    if n <= 9:
        return "GS-8–9"
    if n <= 11:
        return "GS-10–11"
    return "GS-12+"


# staircase in the collapsed grid: education rows a degree could qualify for,
# per grade group (a higher degree also covers the lower groups).
SIMPLE_QUAL = {
    "≤ GS-7":   {"Bachelor's / post-bach.", "Master's / professional", "Doctorate"},
    "GS-8–9":   {"Master's / professional", "Doctorate"},
    "GS-10–11": {"Doctorate"},
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


# pay-plan scope -> SQL filter on pay_plan_code
PLAN_FILTERS = {
    "gs": "AND pay_plan_code = 'GS'",            # General Schedule only
    "gs+gg": "AND pay_plan_code IN ('GS', 'GG')", # GS + GG (same grade scale)
    "all": "",                                    # every pay plan
}


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
        name_row = con.execute(
            f"SELECT occupational_series FROM {src} WHERE {where} "
            f"GROUP BY 1 ORDER BY SUM(TRY_CAST(count AS BIGINT)) DESC LIMIT 1"
        ).fetchone()
        name = name_row[0] if name_row else series
    df = con.execute(f"""
        SELECT pay_plan_code, grade,
               education_level_code AS edu,
               SUM(TRY_CAST(count AS BIGINT)) AS hires
        FROM {src}
        WHERE {where} {PLAN_FILTERS[pay_plans]}
        GROUP BY 1, 2, 3
    """).df()
    return df, name


def _matrix(df, pay_plans, simple=False):
    df = df.copy()
    df["bucket"] = df["edu"].map(lambda c: CODE_TO_BUCKET.get(c, "Below HS / unknown"))
    index = EDU_ORDER
    if pay_plans == "all":
        # GS & GG share the GS grade scale; everything else -> one pooled column
        gsgg = df["pay_plan_code"].isin(["GS", "GG"]) & df["grade"].isin(GRADE_ORDER)
        df["col"] = np.where(gsgg, df["grade"], OTHER_COL)
        cols = GRADE_ORDER + [OTHER_COL]
    elif simple:
        # collapse to 5 education rows x 4 grade groups (empty low grades fold in)
        df = df[df["grade"].isin(GRADE_ORDER)]
        df["bucket"] = df["bucket"].map(SIMPLE_EDU_MAP)
        df["col"] = df["grade"].map(_grade_group)
        index, cols = SIMPLE_EDU_ORDER, SIMPLE_GRADE_ORDER
    else:
        # gs / gs+gg: already filtered to comparable plans, just the grade scale
        df = df[df["grade"].isin(GRADE_ORDER)]
        df["col"] = df["grade"]
        cols = GRADE_ORDER
    piv = (df.groupby(["bucket", "col"])["hires"].sum().reset_index()
             .pivot(index="bucket", columns="col", values="hires"))
    return piv.reindex(index=index, columns=cols).fillna(0)


# --------------------------------------------------------------------------- #
# the one public function
# --------------------------------------------------------------------------- #
_SUFFIX = {"gs": "", "gs+gg": "_gsgg", "all": "_allplans"}


def accession_heatmap(series=None, pay_plans=None, all_plans=False,
                      totals=True, highlight_quals=False, simple=False,
                      save=False, out=None):
    """Render the grade x education heatmap of new federal hires.

    Parameters
    ----------
    series : str or None
        Occupational series code, e.g. "2210" (IT Mgmt), "1550" (Comp Sci).
        None -> ALL occupations combined (overall hiring).
    pay_plans : {"gs", "gs+gg", "all"} or None
        Which pay plans to include and how to lay out the x-axis:
          "gs"    -> General Schedule only (GS-01..GS-15).
          "gs+gg" -> GS and GG together on the GS grade scale (no other column).
          "all"   -> GS+GG on the grade scale; every other pay plan pooled into
                     a single "Other plans" column.
        Default None resolves to "all" if all_plans=True, else "gs".
    all_plans : bool
        Backward-compatible shortcut: True == pay_plans="all".
    totals : bool   show the grey row/column total strips (default True).
    highlight_quals : bool
        Outline cells where a hire's degree could qualify them for that grade.
        Staircase: bachelor's opens GS-5/7, master's GS-5/7/9, Ph.D. GS-5/7/9/11
        (each degree also covers the lower degree-entry grades).
    simple : bool
        Readable Substack layout: 5 education rows (≤HS, some college/assoc,
        bachelor's/post-bach, master's/professional, doctorate) x 4 grade groups
        (≤GS-7, GS-8–9, GS-10–11, GS-12+). Totals show % only. Implies GS+GG.
    save : bool   write a PNG (default heatmap_<series|all>[_gsgg|_allplans].png)
    out : str     explicit output path (implies save)

    Returns
    -------
    matplotlib.figure.Figure  (displays inline in a notebook)
    """
    if pay_plans is None:
        pay_plans = "gs+gg" if simple else ("all" if all_plans else "gs")
    pay_plans = pay_plans.lower().replace("gsgg", "gs+gg")
    if pay_plans not in PLAN_FILTERS:
        raise ValueError(f"pay_plans must be one of {list(PLAN_FILTERS)}")
    if simple and pay_plans == "all":
        pay_plans = "gs+gg"          # the pooled column has no place in the grouped grid
    series = None if series is None else str(series)

    df, series_name = _fetch(series, pay_plans)
    mat = _matrix(df, pay_plans, simple=simple)
    quals = SIMPLE_QUAL if simple else QUAL_RULES
    fig = _plot(mat, series, series_name, pay_plans, quals,
                show_totals=totals, highlight_quals=highlight_quals, simple=simple)
    if save or out:
        tag = series if series is not None else "all"
        suffix = _SUFFIX[pay_plans] + ("_simple" if simple else "")
        path = out or f"heatmap_{tag}{suffix}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        print(f"wrote {path}")
    return fig


def _plot(mat, series, series_name, pay_plans, quals=QUAL_RULES,
          show_totals=True, highlight_quals=False, simple=False):
    pooled = pay_plans == "all"      # has an "Other plans" column
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#cccccc",
        "figure.facecolor": "white",
    })
    data = mat.values
    nrows, ncols = data.shape
    grand = data.sum()
    col_tot = data.sum(axis=0)
    row_tot = data.sum(axis=1)
    vmax = data.max() or 1
    col_labels = [str(c) if simple else (c if c == OTHER_COL else f"GS-{c}")
                  for c in mat.columns]
    cell_fs = 11 if simple else 7.5
    tick_fs = 11 if simple else 9

    if simple:
        fig, ax = plt.subplots(figsize=(2.6 + 1.7 * ncols, 1.9 + 0.74 * nrows))
    else:
        fig, ax = plt.subplots(figsize=(max(13.5, 1.5 + 0.78 * ncols), 7))
    im = ax.imshow(data, aspect="auto", cmap=CMAP, vmin=0, vmax=vmax)

    # crisp white gridlines, no spines/ticks
    ax.set_xticks(np.arange(-.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, nrows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    ax.set_xticks(range(ncols))
    ax.set_xticklabels(col_labels, fontsize=tick_fs, rotation=90 if pooled else 0)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(list(mat.index), fontsize=tick_fs)
    # Label top+bottom only when there's room: the pooled view's vertical
    # "Other plans" label and the qualifying-cells caption both crowd the top.
    top_labels = not pooled and not highlight_quals
    ax.tick_params(axis="x", labeltop=top_labels, labelbottom=True)

    def fmt(v):
        return f"{int(v):,}" if v else ""

    def fmt_pct(v):
        if not (v and grand):
            return ""
        return f"{round(100*v/grand)}%" if simple else f"{100*v/grand:.1f}%"

    # cells a hire's degree could qualify them for -> bold outline
    qual_color = "#1565c0"
    active = quals if highlight_quals else {}
    qual_cells = set()
    for j, c in enumerate(mat.columns):
        for i, bucket in enumerate(mat.index):
            if c in active and bucket in active[c] and data[i, j] > 0:
                qual_cells.add((i, j))

    # body cell counts (bold for highlighted cells)
    for i in range(nrows):
        for j in range(ncols):
            v = data[i, j]
            if not v:
                continue
            hot = (i, j) in qual_cells
            ax.text(j, i, fmt(v), ha="center", va="center",
                    fontsize=cell_fs + 0.5 if hot else cell_fs,
                    fontweight="bold" if hot else "normal",
                    color="white" if v > 0.55 * vmax else "#333333")

    for (i, j) in qual_cells:
        ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False,
                               edgecolor=qual_color, lw=2.5, zorder=5))

    # callout box with the headline share
    if highlight_quals and qual_cells:
        qual_sum = sum(data[i, j] for (i, j) in qual_cells)
        share = qual_sum / grand if grand else 0
        if simple:
            # below the chart, centered (the grouped grid has no empty corner)
            ax.annotate(
                f"{share:.0%} of hires (n = {int(qual_sum):,}) sit in an outlined cell — "
                "their degree could qualify them for that grade",
                xy=(0.5, 0), xycoords="axes fraction", xytext=(0, -48),
                textcoords="offset points", ha="center", va="top",
                fontsize=10, color="#0d3b66",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                          edgecolor=qual_color, linewidth=1.8))
        else:
            # into the empty low-grade corner
            ax.text(
                -0.3, 2.1,
                f"{share:.0%} of hires\n(n = {int(qual_sum):,})\nsit in an outlined cell —\n"
                "their degree could qualify\nthem for that grade",
                ha="left", va="center", fontsize=9.5, color="#0d3b66", zorder=6,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                          edgecolor=qual_color, linewidth=1.8, alpha=0.95))

    # margin totals (grey strips). simple: % only; otherwise count + %
    if show_totals:
        grey, dark = "#e8e8e8", "#bdbdbd"
        for i in range(nrows):
            ax.add_patch(Rectangle((ncols - .5, i - .5), 1, 1, facecolor=grey,
                                   edgecolor="white", lw=2, clip_on=False))
            if row_tot[i] and simple:
                ax.text(ncols, i, fmt_pct(row_tot[i]), ha="center", va="center",
                        fontsize=10, fontweight="bold", color="#222222")
            elif row_tot[i]:
                ax.text(ncols, i - .15, fmt(row_tot[i]), ha="center", va="center",
                        fontsize=8.5, fontweight="bold", color="#222222")
                ax.text(ncols, i + .22, fmt_pct(row_tot[i]), ha="center", va="center",
                        fontsize=7.5, color="#777777")
        for j in range(ncols):
            ax.add_patch(Rectangle((j - .5, nrows - .5), 1, 1, facecolor=grey,
                                   edgecolor="white", lw=2, clip_on=False))
            if col_tot[j] and simple:
                ax.text(j, nrows, fmt_pct(col_tot[j]), ha="center", va="center",
                        fontsize=10, fontweight="bold", color="#222222")
            elif col_tot[j]:
                ax.text(j, nrows - .15, fmt(col_tot[j]), ha="center", va="center",
                        fontsize=8.5, fontweight="bold", color="#222222")
                ax.text(j, nrows + .22, fmt_pct(col_tot[j]), ha="center", va="center",
                        fontsize=7.5, color="#777777")
        ax.add_patch(Rectangle((ncols - .5, nrows - .5), 1, 1, facecolor=dark,
                               edgecolor="white", lw=2, clip_on=False))
        ax.text(ncols, nrows, "100%" if simple else fmt(grand), ha="center",
                va="center", fontsize=9, fontweight="bold", color="black")
        if not simple:   # simple mode: the grey % strips are self-evident
            ax.text(ncols, -1.0, "Total", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="#555555")
        ax.text(-0.85, nrows, "%" if simple else "Total", ha="right", va="center",
                fontsize=9, fontweight="bold", color="#555555")

    pad = .5 if show_totals else -.5
    ax.set_xlim(-.5, ncols + pad)
    ax.set_ylim(nrows + pad, -.5)
    xlabel = "Grade group  (GS + GG)" if simple else {
        "gs": "Grade",
        "gs+gg": "Grade  (GS + GG)",
        "all": "Grade  (GS + GG; all other pay plans pooled as 'Other plans')",
    }[pay_plans]
    ax.set_xlabel(xlabel, fontsize=10, labelpad=8)
    ax.set_ylabel("Education level", fontsize=10)

    title_name = series_name.title() if series_name.isupper() else series_name
    if simple:
        prefix = "All occupations" if series is None else f"{series}  {title_name}"
        ax.set_title(prefix, fontsize=14, fontweight="bold", pad=42, loc="left")
        plans_label = "GS + GG" if pay_plans == "gs+gg" else "GS"
        ax.annotate(
            f"New {plans_label} hires by grade & education   ·   {_date_range()}   "
            f"·   n = {int(grand):,}   ·   OPM / EHRI",
            xy=(0, 1), xycoords="axes fraction", xytext=(0, 24),
            textcoords="offset points", fontsize=9.5, color="#666666", ha="left")
    else:
        scope = {
            "gs": "new GS hires",
            "gs+gg": "new GS + GG hires",
            "all": "new hires, all pay plans",
        }[pay_plans]
        prefix = title_name if series is None else f"{series} ({title_name})"
        ax.set_title(f"{prefix} — {scope} by grade & education",
                     fontsize=15, fontweight="bold", pad=34, loc="left")
        ax.annotate(
            f"{_date_range()}   ·   n = {int(grand):,}   ·   source: OPM / EHRI accessions",
            xy=(0, 1), xycoords="axes fraction", xytext=(0, 22),
            textcoords="offset points", fontsize=10, color="#666666", ha="left")
    if highlight_quals:
        caption = ("Outlined: a hire's degree could qualify them for that grade "
                   "(bachelor's→GS-7, master's→GS-9, Ph.D.→GS-11)") if simple else (
            "Outlined cells: grades a hire's degree could qualify them for "
            "(OPM: bachelor's→GS-5/7, master's→GS-9, Ph.D.→GS-11; a degree also qualifies for grades below it)")
        ax.annotate(caption, xy=(0, 1), xycoords="axes fraction", xytext=(0, 8),
                    textcoords="offset points", fontsize=8.5, color="#1565c0", ha="left")

    # no colorbar: every cell is labeled with its count, so the scale is redundant
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    # usage: python ehri_heatmaps.py [SERIES|overall] [gs|gs+gg|allplans]
    import sys
    series, pay_plans = "2210", "gs"
    for a in sys.argv[1:]:
        al = a.lower()
        if al in ("gs", "gsgg", "gs+gg", "all", "allplans"):
            pay_plans = {"gsgg": "gs+gg", "allplans": "all"}.get(al, al)
        elif al in ("overall", "all-series", "none"):
            series = None
        else:
            series = a
    accession_heatmap(series, pay_plans=pay_plans, save=True)
