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
import functools

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import duckdb
from huggingface_hub import HfApi

REPO = "impactproject/opm-ehri-data"
START_MONTH = "202101"          # earliest accession month included
OTHER_COL = "Other plans"       # all-plans mode: pool of non-GS/GG hires

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


def _matrix(df, pay_plans):
    df = df.copy()
    df["bucket"] = df["edu"].map(lambda c: CODE_TO_BUCKET.get(c, "Below HS / unknown"))
    if pay_plans == "all":
        # GS & GG share the GS grade scale; everything else -> one pooled column
        gsgg = df["pay_plan_code"].isin(["GS", "GG"]) & df["grade"].isin(GRADE_ORDER)
        df["col"] = np.where(gsgg, df["grade"], OTHER_COL)
        cols = GRADE_ORDER + [OTHER_COL]
    else:
        # gs / gs+gg: already filtered to comparable plans, just the grade scale
        df = df[df["grade"].isin(GRADE_ORDER)]
        df["col"] = df["grade"]
        cols = GRADE_ORDER
    piv = (df.groupby(["bucket", "col"])["hires"].sum().reset_index()
             .pivot(index="bucket", columns="col", values="hires"))
    return piv.reindex(index=EDU_ORDER, columns=cols).fillna(0)


# --------------------------------------------------------------------------- #
# the one public function
# --------------------------------------------------------------------------- #
_SUFFIX = {"gs": "", "gs+gg": "_gsgg", "all": "_allplans"}


def accession_heatmap(series=None, pay_plans=None, all_plans=False,
                      totals=True, highlight_quals=False, save=False, out=None):
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
        Outline cells where the degree alone could qualify the hire for that
        grade. Staircase: bachelor's opens GS-5/7, master's GS-5/7/9, Ph.D.
        GS-5/7/9/11 (each degree also covers the lower degree-entry grades).
    save : bool   write a PNG (default heatmap_<series|all>[_gsgg|_allplans].png)
    out : str     explicit output path (implies save)

    Returns
    -------
    matplotlib.figure.Figure  (displays inline in a notebook)
    """
    if pay_plans is None:
        pay_plans = "all" if all_plans else "gs"
    pay_plans = pay_plans.lower().replace("gsgg", "gs+gg")
    if pay_plans not in PLAN_FILTERS:
        raise ValueError(f"pay_plans must be one of {list(PLAN_FILTERS)}")
    series = None if series is None else str(series)

    df, series_name = _fetch(series, pay_plans)
    mat = _matrix(df, pay_plans)
    fig = _plot(mat, series, series_name, pay_plans,
                show_totals=totals, highlight_quals=highlight_quals)
    if save or out:
        tag = series if series is not None else "all"
        path = out or f"heatmap_{tag}{_SUFFIX[pay_plans]}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        print(f"wrote {path}")
    return fig


def _plot(mat, series, series_name, pay_plans, show_totals=True,
          highlight_quals=False):
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
    col_labels = [c if c == OTHER_COL else f"GS-{c}" for c in mat.columns]
    width = max(13.5, 1.5 + 0.78 * ncols)

    fig, ax = plt.subplots(figsize=(width, 7))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)

    # crisp white gridlines, no spines/ticks
    ax.set_xticks(np.arange(-.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, nrows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    ax.set_xticks(range(ncols))
    ax.set_xticklabels(col_labels, fontsize=9, rotation=90 if pooled else 0)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(EDU_ORDER, fontsize=9)
    # Label top+bottom only when there's room: the pooled view's vertical
    # "Other plans" label and the qualifying-cells caption both crowd the top.
    top_labels = not pooled and not highlight_quals
    ax.tick_params(axis="x", labeltop=top_labels, labelbottom=True)

    def fmt(v):
        return f"{int(v):,}" if v else ""

    def fmt_pct(v):
        return f"{100*v/grand:.1f}%" if v and grand else ""

    # cells where the degree could be the qualifying credential -> bold outline
    qual_color = "#1565c0"
    quals = QUAL_RULES if highlight_quals else {}
    qual_cells = set()
    for j, c in enumerate(mat.columns):
        for i, bucket in enumerate(EDU_ORDER):
            if c in quals and bucket in quals[c] and data[i, j] > 0:
                qual_cells.add((i, j))

    # body cell counts (bold for highlighted cells)
    for i in range(nrows):
        for j in range(ncols):
            v = data[i, j]
            if not v:
                continue
            hot = (i, j) in qual_cells
            ax.text(j, i, fmt(v), ha="center", va="center",
                    fontsize=8 if hot else 7.5,
                    fontweight="bold" if hot else "normal",
                    color="white" if v > 0.55 * vmax else "#333333")

    for (i, j) in qual_cells:
        ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False,
                               edgecolor=qual_color, lw=2.5, zorder=5))

    # callout box with the headline share, dropped into the empty low-grade corner
    if highlight_quals and qual_cells:
        qual_sum = sum(data[i, j] for (i, j) in qual_cells)
        share = qual_sum / grand if grand else 0
        ax.text(
            -0.3, 2.1,
            f"{share:.0%} of hires\n(n = {int(qual_sum):,})\nsit in an outlined cell —\n"
            "their degree alone could\nqualify them for that grade",
            ha="left", va="center", fontsize=9.5, color="#0d3b66", zorder=6,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor=qual_color, linewidth=1.8, alpha=0.95))

    # margin totals (grey strips): count + share of all hires
    if show_totals:
        grey, dark = "#e8e8e8", "#bdbdbd"
        for i in range(nrows):
            ax.add_patch(Rectangle((ncols - .5, i - .5), 1, 1, facecolor=grey,
                                   edgecolor="white", lw=2, clip_on=False))
            if row_tot[i]:
                ax.text(ncols, i - .15, fmt(row_tot[i]), ha="center", va="center",
                        fontsize=8.5, fontweight="bold", color="#222222")
                ax.text(ncols, i + .22, fmt_pct(row_tot[i]), ha="center", va="center",
                        fontsize=7.5, color="#777777")
        for j in range(ncols):
            ax.add_patch(Rectangle((j - .5, nrows - .5), 1, 1, facecolor=grey,
                                   edgecolor="white", lw=2, clip_on=False))
            if col_tot[j]:
                ax.text(j, nrows - .15, fmt(col_tot[j]), ha="center", va="center",
                        fontsize=8.5, fontweight="bold", color="#222222")
                ax.text(j, nrows + .22, fmt_pct(col_tot[j]), ha="center", va="center",
                        fontsize=7.5, color="#777777")
        ax.add_patch(Rectangle((ncols - .5, nrows - .5), 1, 1, facecolor=dark,
                               edgecolor="white", lw=2, clip_on=False))
        ax.text(ncols, nrows, fmt(grand), ha="center", va="center",
                fontsize=9, fontweight="bold", color="black")
        ax.text(ncols, -1.0, "Total", ha="center", va="center",
                fontsize=9, fontweight="bold", color="#555555")
        ax.text(-0.85, nrows, "Total", ha="right", va="center",
                fontsize=9, fontweight="bold", color="#555555")

    pad = .5 if show_totals else -.5
    ax.set_xlim(-.5, ncols + pad)
    ax.set_ylim(nrows + pad, -.5)
    xlabel = {
        "gs": "Grade",
        "gs+gg": "Grade  (GS + GG)",
        "all": "Grade  (GS + GG; all other pay plans pooled as 'Other plans')",
    }[pay_plans]
    ax.set_xlabel(xlabel, fontsize=10, labelpad=8)
    ax.set_ylabel("Education level", fontsize=10)

    title_name = series_name.title() if series_name.isupper() else series_name
    scope = {
        "gs": "new GS hires",
        "gs+gg": "new GS + GG hires",
        "all": "new hires, all pay plans",
    }[pay_plans]
    prefix = title_name if series is None else f"{series} ({title_name})"
    ax.set_title(f"{prefix} — {scope} by grade & education",
                 fontsize=15, fontweight="bold", pad=34, loc="left")
    ax.annotate(
        f"Jan 2021 – present   ·   n = {int(grand):,}   ·   source: OPM / EHRI accessions",
        xy=(0, 1), xycoords="axes fraction", xytext=(0, 22),
        textcoords="offset points", fontsize=10, color="#666666", ha="left")
    if highlight_quals:
        ax.annotate(
            "Outlined cells: grades the degree alone could qualify the hire for "
            "(OPM: bachelor's→GS-5/7, master's→GS-9, Ph.D.→GS-11; a degree also qualifies for grades below it)",
            xy=(0, 1), xycoords="axes fraction", xytext=(0, 8),
            textcoords="offset points", fontsize=8.5, color="#1565c0", ha="left")

    cb = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.06)
    cb.set_label("New hires", fontsize=9)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=8)
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
