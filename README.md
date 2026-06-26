# Federal new-hire heatmaps: grade × education by occupation

Reproducible heatmaps of **new federal hires** by **GS grade** and **education
level**, for any occupational series — e.g. 2210 (IT Management), 1550 (Computer
Science), 1560 (Data Science), 1530 (Statistics).

One clean, publication-ready layout — **5 education buckets × 4 grade groups** —
off one function:

```python
from ehri_heatmaps import accession_heatmap

accession_heatmap("2210")               # IT Management
accession_heatmap()                     # all occupations
accession_heatmap("1530", pay_plans="gs")   # GS only (default is GS + GG)
```

Defaults: GS + GG, with the degree→grade *could-qualify* staircase outlined
(bachelor's→GS-7, master's→GS-9, Ph.D.→GS-11) and the headline share in a box.
Options: `series` (a code, or `None` for all occupations) · `pay_plans`
(`"gs+gg"` default, or `"gs"`) · `highlight_quals` (default `True`) ·
`totals` (the % margin strips, default `True`).

Open [`heatmaps.ipynb`](heatmaps.ipynb) to see a figure rendered for each series.
Its first half walks the data pull step by step so nothing is a black box.

### See the data behind a figure

The same numbers a chart draws are available as plain tables:

```python
ehri.dataset_info()       # dataset, # monthly files, date range
ehri.build_sql("2210")    # the exact DuckDB query (readable form)
ehri.fetch_counts("2210") # raw query result: hires by (grade, education code)
ehri.bucket_maps()        # how codes/grades collapse into the 5x4 buckets
ehri.crosstab("2210")     # the 5x4 table of hires the heatmap colors
```

## Data

Source: the public HuggingFace dataset
[`impactproject/opm-ehri-data`](https://huggingface.co/datasets/impactproject/opm-ehri-data),
which mirrors OPM/EHRI **accessions** (new federal hires) published at
[data.opm.gov](https://data.opm.gov/explore-data/data/data-downloads).

- Parquet is **streamed remotely with DuckDB** — nothing is downloaded to disk.
- Each source row is pre-aggregated, so counts are `SUM(count)`, not row counts.
- Scope per figure: a single `occupational_series_code` (or all occupations),
  accession effective month ≥ `START_MONTH` (Jan 2021 by default; change it in
  `ehri_heatmaps.py`).

### Buckets

- **Education (5):** ≤ high school · some college/associate · bachelor's/post-bach
  · master's/professional · doctorate. "Master's / professional" folds in
  first-professional and sixth-year degrees, credited at GS-9.
- **Grade groups (4):** ≤ GS-7 · GS-8–9 · GS-10–11 · GS-12+, on the GS scale
  (GS + GG by default; `pay_plans="gs"` for GS only).
- **Qualifying staircase:** outlined cells are where a degree could qualify the
  hire for that grade (bachelor's→GS-7, master's→GS-9, Ph.D.→GS-11; a higher
  degree also covers the lower groups).

## Reproduce

```bash
pip install -r requirements.txt
jupyter lab heatmaps.ipynb      # run all cells
# or regenerate a single PNG from the CLI:
python ehri_heatmaps.py 2210           # -> heatmap_2210.png
python ehri_heatmaps.py overall gs     # all occupations, GS only -> heatmap_all.png
```

## Citing

These figures are generated from public OPM/EHRI data; cite both this repository
and the underlying dataset. For a durable citation, archive a tagged release to
[Zenodo](https://zenodo.org/) and cite the resulting DOI.

## License

Code released under the MIT License (see `LICENSE`). The underlying OPM/EHRI data
is U.S. Government work in the public domain.
