# Daytime LST decline over Ajmer (NW India): analysis code and data

This repository contains the code and input data that reproduce all numbers, tables and figures in the manuscript
**"A Two-Decade Decline in Daytime Land Surface Temperature over Ajmer, Semi-Arid Northwest India: Sensor Consistency,
Reanalysis Corroboration, and Attribution to Post-Monsoon Wetting and Delayed Canopy Senescence."**

The code:

1. computes Theil–Sen / Mann–Kendall trends (2001–2021) in MODIS Terra and Aqua day and night LST over a 17 × 17 km window centred on Ajmer (Table 1);
2. tests the trend against overpass-time drift, view-zenith-angle drift, clear-sky sampling and QA pass rate (§3.2, Fig. 2);
3. compares MODIS with ERA5-Land skin temperature interpolated to each scene's local solar overpass time (Table 2, Fig. 3);
4. resolves the trend by calendar month (Fig. 4), tests the 2003–2012 vs 2013–2021 step, and checks VIIRS VNP21A2;
5. computes MCD12Q2 phenology trends (Table 3) and the attribution statistics linking post-monsoon LST to senescence timing and soil moisture (§3.5–3.6, Fig. 5).

## Repository structure

```
README.md
requirements.txt
src/
  lstlib.py                  statistics and I/O helpers (Theil-Sen, Mann-Kendall, anomalies, interpolation)
  ajmer_lst_analysis.py      full analysis -> results/*.csv and results/summary.txt
  ajmer_lst_figures.py       Figures 1-5 (400 dpi PNG, numbered as in the manuscript) -> results/
tests/
  test_quick.py              quick test, a few seconds, uses only the example file below
  example_data/synthetic_lst_8day.csv   synthetic 8-day LST series with a known -1.5 K/decade trend
  test_reproduce.py          full reproduction check against expected_results/ (~15 s)
expected_results/            reference outputs (Tables 1-3 and the full summary log)
data/                        all input data as individual CSV files (see below)
```

## Requirements

Python 3.9 or later with the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

Tested with Python 3.12, numpy 2.4, pandas 3.0, scipy 1.17 and matplotlib 3.10.

## How to run the quick test

From the repository root:

```
python tests/test_quick.py
```

This finishes in a few seconds and does not need the project data. It reads `tests/example_data/synthetic_lst_8day.csv`
(an 8-day series with a seasonal cycle, noise and a built-in trend of −1.5 K/decade), applies the same
month-anomaly → annual-mean → Theil–Sen procedure used in the paper, and checks that the known trend is recovered.
It also checks the clear-sky bit counting, the hourly interpolation and the partial-correlation helper.
Expected ending: `Ran 5 tests ... OK`.

## How to reproduce the paper

From the repository root:

```
python src/ajmer_lst_analysis.py      # about 10 s; writes results/summary.txt and results/*.csv
python src/ajmer_lst_figures.py       # about 5 s; writes results/fig1 ... fig5 PNG files
```

To confirm your run matches the reference outputs:

```
python tests/test_reproduce.py        # Expected ending: Ran 3 tests ... OK
```

| Output file | Manuscript item |
|---|---|
| `results/table1_trends.csv` | Table 1 |
| `results/table2_scene_comparison.csv` | Table 2 |
| `results/table3_phenology.csv` | Table 3 |
| `results/geometry_controls.csv`, `summary.txt` (§3.2 block) | §3.2 |
| `results/month_resolved_trends.csv` | §3.4 |
| `results/era5_hydroclimate_annual.csv`, `summary.txt` (attribution block) | §3.5–3.6 |
| `results/fig1_lst_timeseries.png` | Figure 1 |
| `results/fig2_artifact_controls.png` | Figure 2 |
| `results/fig3_era5_step.png` | Figure 3 |
| `results/fig4_month_resolved.png` | Figure 4 |
| `results/fig5_phenology_moisture.png` | Figure 5 |

## Data

All inputs are publicly available; the subsets used are included in `data/` as individual CSV files.

| Folder | Product | Source |
|---|---|---|
| `data/MOD11A2/` | MODIS Terra LST 8-day 1 km, Collection 6 (pixel subset + scene statistics) | ORNL DAAC Global Subsetting Tool, https://doi.org/10.3334/ORNLDAAC/1379; product https://doi.org/10.5067/MODIS/MOD11A2.006 |
| `data/MYD11A2/` | MODIS Aqua LST 8-day 1 km, Collection 6 | as above; https://doi.org/10.5067/MODIS/MYD11A2.006 |
| `data/VNP21A2/` | VIIRS LST 8-day 1 km, V001 (scene statistics) | as above; https://doi.org/10.5067/VIIRS/VNP21A2.001 |
| `data/MCD12Q1/` | MODIS land cover 2019, LC_Type1, resampled to the 1 km window | as above |
| `data/MCD12Q2/` | MODIS land surface phenology 2001–2019, Collection 6 | as above; https://doi.org/10.5067/MODIS/MCD12Q2.006 |
| `data/ERA5Land/hourly/` | ERA5-Land hourly skin temperature and 2 m temperature at the UTC hours bracketing each overpass (05, 06, 08, 09, 17, 18, 20, 21 UTC), 2000–2022 | Google Earth Engine `ECMWF/ERA5_LAND/HOURLY` |
| `data/ERA5Land/era5land_daily_ajmer.csv` | ERA5-Land daily precipitation, soil water layers 1–2, skin/2 m temperature, transpiration, total evaporation, LAI | Google Earth Engine `ECMWF/ERA5_LAND/DAILY_AGGR` |

Notes on the data:

- ORNL subset centres: the Terra MOD11A2 subset is centred at 26.47057° N, 74.64105° E. The Aqua MYD11A2 and VIIRS VNP21A2 subsets are centred at 26.44598° N, 74.63502° E, about 2.7 km south of the Terra window; the two 17 × 17 km windows overlap by about 81%.
- The ERA5-Land files were extracted with Google Earth Engine; the extraction script is not included. The CSV files are provided exactly as extracted.
- ERA5-Land data: Copernicus Climate Change Service; use is subject to the Copernicus licence, which requires attribution.

## Methodological choices fixed in the code

- Trends use complete calendar years only: Terra 2001–2021; Aqua 2003–2021 (Aqua begins 4 July 2002, so 2002 is a partial year and is excluded from all Aqua analyses).
- Scene means outside 250–340 K are excluded. Anomalies are taken from the calendar-month climatology before annual averaging.
- Mann–Kendall p-values use the normal approximation with continuity and tie corrections; confidence intervals are 95% Theil–Sen intervals.
- Clear-sky count per composite is the number of set bits in the MODIS `Clear_sky_days` / `Clear_sky_nights` 8-bit mask, averaged over the window.
- Monsoon withdrawal (§3.5) is defined as the day of year by which 95% of June–October ERA5-Land rainfall has accumulated.

## License

To be added.

## Citation

If you use this code, please cite the associated article (reference to be added on publication).
