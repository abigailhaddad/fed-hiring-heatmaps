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


@functools.lru_cache(maxsize=64)
def _fetch(series, all_plans):
    con = _connection()
    lst = "[" + ",".join(f"'{u}'" for u in _monthly_urls()) + "]"
    src = f"read_parquet({lst})"
    where = (f"occupational_series_code = '{series}' "
             f"AND personnel_action_effective_date_yyyymm >= '{START_MONTH}'")
    name_row = con.execute(
        f"SELECT occupational_series FROM {src} WHERE {where} "
        f"GROUP BY 1 ORDER BY SUM(TRY_CAST(count AS BIGINT)) DESC LIMIT 1"
    ).fetchone()
    name = name_row[0] if name_row else series
    plan_filter = "" if all_plans else "AND pay_plan_code = 'GS'"
    df = con.execute(f"""
        SELECT pay_plan_code, grade,
               education_level_code AS edu,
               SUM(TRY_CAST(count AS BIGINT)) AS hires
        FROM {src}
        WHERE {where} {plan_filter}
        GROUP BY 1, 2, 3
    """).df()
    return df, name


def _matrix(df, all_plans):
    df = df.copy()
    df["bucket"] = df["edu"].map(lambda c: CODE_TO_BUCKET.get(c, "Below HS / unknown"))
    if not all_plans:
        df = df[df["grade"].isin(GRADE_ORDER)]
        cols = GRADE_ORDER
        df["col"] = df["grade"]
    else:
        # GS & GG share the GS grade scale; everything else -> one pooled column
        gsgg = df["pay_plan_code"].isin(["GS", "GG"]) & df["grade"].isin(GRADE_ORDER)
        df["col"] = np.where(gsgg, df["grade"], OTHER_COL)
        cols = GRADE_ORDER + [OTHER_COL]
    piv = (df.groupby(["bucket", "col"])["hires"].sum().reset_index()
             .pivot(index="bucket", columns="col", values="hires"))
    return piv.reindex(index=EDU_ORDER, columns=cols).fillna(0)


# --------------------------------------------------------------------------- #
# the one public function
# --------------------------------------------------------------------------- #
def accession_heatmap(series, all_plans=False, save=False, out=None):
    """Render the grade x education heatmap for an occupational series.

    Parameters
    ----------
    series : str   e.g. "2210" (IT Mgmt), "1550" (Comp Sci), "1530" (Statistics)
    all_plans : bool
        False -> GS pay plan only (x-axis = GS-01..GS-15).
        True  -> GS+GG share the grade scale; all other pay plans (demo bands,
                 AD, etc.) are pooled into a single "Other plans" column.
    save : bool   if True, write a PNG (default name heatmap_<series>[_allplans].png)
    out : str     explicit output path (implies save)

    Returns
    -------
    matplotlib.figure.Figure  (displays inline in a notebook)
    """
    series = str(series)
    df, series_name = _fetch(series, all_plans)
    mat = _matrix(df, all_plans)
    fig = _plot(mat, series, series_name, all_plans)
    if save or out:
        suffix = "_allplans" if all_plans else ""
        path = out or f"heatmap_{series}{suffix}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        print(f"wrote {path}")
    return fig


def _plot(mat, series, series_name, all_plans):
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
    ax.set_xticklabels(col_labels, fontsize=9, rotation=90 if all_plans else 0)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(EDU_ORDER, fontsize=9)
    # GS mode is compact enough to label top+bottom; all-plans bottom only
    # (vertical labels would collide with the subtitle).
    ax.tick_params(axis="x", labeltop=not all_plans, labelbottom=True)

    def fmt(v):
        return f"{int(v):,}" if v else ""

    def fmt_pct(v):
        return f"{100*v/grand:.1f}%" if v and grand else ""

    # body cell counts
    for i in range(nrows):
        for j in range(ncols):
            v = data[i, j]
            if not v:
                continue
            ax.text(j, i, fmt(v), ha="center", va="center", fontsize=7.5,
                    color="white" if v > 0.55 * vmax else "#333333")

    # margin totals (grey strips): count + share of all hires
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

    ax.set_xlim(-.5, ncols + .5)
    ax.set_ylim(nrows + .5, -.5)
    ax.set_xlabel("Grade  (GS + GG; all other pay plans pooled as 'Other plans')"
                  if all_plans else "Grade", fontsize=10, labelpad=8)
    ax.set_ylabel("Education level", fontsize=10)

    title_name = series_name.title() if series_name.isupper() else series_name
    scope = "new hires, all pay plans" if all_plans else "new GS hires"
    ax.set_title(f"{series} ({title_name}) — {scope} by grade & education",
                 fontsize=15, fontweight="bold", pad=34, loc="left")
    ax.annotate(
        f"Jan 2021 – present   ·   n = {int(grand):,}   ·   source: OPM / EHRI accessions",
        xy=(0, 1), xycoords="axes fraction", xytext=(0, 22),
        textcoords="offset points", fontsize=10, color="#666666", ha="left")

    cb = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.06)
    cb.set_label("New hires", fontsize=9)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=8)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    ap = "all" in args
    pos = [a for a in args if a != "all"] or ["2210"]
    accession_heatmap(pos[0], all_plans=ap, save=True)
