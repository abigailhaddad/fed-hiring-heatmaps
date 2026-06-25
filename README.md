# Federal new-hire heatmaps: grade × education by occupation

Reproducible heatmaps of **new federal hires** by **GS grade** and **education
level**, for any occupational series — e.g. 2210 (IT Management), 1550 (Computer
Science), 1560 (Data Science), 1530 (Statistics).

Everything runs off one function:

```python
from ehri_heatmaps import accession_heatmap

accession_heatmap("2210")                       # GS hires only
accession_heatmap("1530", all_plans=True)       # GS+GG on the grade scale, other plans pooled
accession_heatmap(pay_plans="gs+gg")            # ALL occupations (series=None), GS+GG
accession_heatmap("2210", pay_plans="gs+gg",    # outline cells where the degree could be
                  totals=False, highlight_quals=True)  # the qualifying basis for the grade
```

Options: `series` (a code, or `None` for all occupations) · `pay_plans`
(`"gs"`, `"gs+gg"`, `"all"`) · `totals` (row/column total strips) ·
`highlight_quals` (outline bachelor's→GS-7 / master's→GS-9 / Ph.D.→GS-11 cells).

Open [`heatmaps.ipynb`](heatmaps.ipynb) to see a figure rendered for each series.

## Data

Source: the public HuggingFace dataset
[`impactproject/opm-ehri-data`](https://huggingface.co/datasets/impactproject/opm-ehri-data),
which mirrors OPM/EHRI **accessions** (new federal hires) published at
[data.opm.gov](https://data.opm.gov/explore-data/data/data-downloads).

- Parquet is **streamed remotely with DuckDB** — nothing is downloaded to disk.
- Each source row is pre-aggregated, so counts are `SUM(count)`, not row counts.
- Scope per figure: a single `occupational_series_code`, accession effective
  month ≥ `START_MONTH` (Jan 2021 by default; change it in `ehri_heatmaps.py`).

### GS-only vs. all-plans

`all_plans=False` (default) shows only General Schedule (GS) hires on the
GS-01…GS-15 scale. `all_plans=True` keeps GS **and GG** (which share the GS
grade scale) and pools every other pay plan — demonstration bands (ND, DP, DB,
NH…), AD, FV, etc. — into one **"Other plans"** column, since their grade
numbers are not comparable to GS grades.

## Reproduce

```bash
pip install -r requirements.txt
jupyter lab heatmaps.ipynb      # run all cells
# or regenerate a single PNG from the CLI:
python ehri_heatmaps.py 2210            # -> heatmap_2210.png
python ehri_heatmaps.py 1530 all        # -> heatmap_1530_allplans.png
```

## Citing

These figures are generated from public OPM/EHRI data; cite both this repository
and the underlying dataset. For a durable citation, archive a tagged release to
[Zenodo](https://zenodo.org/) and cite the resulting DOI.

## License

Code released under the MIT License (see `LICENSE`). The underlying OPM/EHRI data
is U.S. Government work in the public domain.
