# Data

## Datasets

| | Full SG | CCR |
|---|---|---|
| Coverage | every private condominium in Singapore | Core Central Region: postal districts 1, 2, 9, 10, 11 |
| Rent projects / size-tier nodes | 2,497 / 12,485 | 804 / 4,020 |
| Sale projects / size-tier nodes | 2,371 / 11,855 | 693 / 3,465 |
| Rent months | Nov 1999 – Jan 2026 (315) | Nov 1999 – Jan 2026 (315) |
| Sale months | Jan 1995 – Feb 2026 (374) | Jan 1995 – Feb 2026 (374) |
| Observed rent records (before 2020 / from 2020) | 318,447 / 180,299 | 109,077 / 54,826 |
| Observed sale records (before 2020 / from 2020) | 153,970 / 61,975 | 41,333 / 11,671 |
| Projects without coordinates (rent / sale) | 1 / 99 | 0 / 0 |
| Release used | September 2026 | July 2026 |

Each panel row is one condominium project, one **size tier** and one calendar month.
Size tiers SZ1–SZ5 are the quintiles of floor area within a project; a size tier is
the unit that is predicted. A row has an observed price only when a rental contract
(rent) or transaction (sale) exists; otherwise it carries only the imputed price.
About 20% of rent cells and 8% of sale cells are observed.

Targets:

- **Rent**: observed monthly rent per square foot, SGD/sq ft/month.
- **Sale**: observed transaction price per square foot, SGD/sq ft.

Losses and metrics use **observed targets only**. Imputed prices are inputs, never labels.

## The data package

`pgim_condo_graph_data.zip` unzips at the repository root into:

```
data/
  raw/
    full_sg_rent.csv          monthly rent panel, all of Singapore (197 columns)
    full_sg_sale.csv          monthly sale panel, all of Singapore (226 columns)
    ccr_rent.csv              monthly rent panel, CCR
    ccr_sale.csv              monthly sale panel, CCR
    project_location.csv      project name -> latitude, longitude, postal district
    rail_transport.csv        MRT/LRT stations with line codes and opening dates
  processed/
    full_sg_rent/  full_sg_sale/  ccr_rent/  ccr_sale/
      node_id.csv             graph node ids: one per project (size_tier 0) and per size tier
      project_info.csv        project name, coordinates, planning area (used by the demo)
      timestep_info.csv       timestep -> month
      project_level_node.csv  project features per month
      size_level_node.csv     size-tier features, targets and masks per month
      project_level_node/ size_level_node/   normalisation statistics (json)
      edges/                  static relation tables (see docs/model.md)
demo/
  predictions/                test predictions of the 12 checkpoints and Full SG
                              monthly graph snapshots, in the demo's format
```

Original file names, for tracing back to the files supplied:

| Here | Supplied as |
|---|---|
| `data/raw/full_sg_rent.csv`, `full_sg_sale.csv` | `R260903.csv`, `S260903.csv` |
| `data/raw/ccr_rent.csv`, `ccr_sale.csv` | `R260713.csv`, `S260713.csv` |
| `data/raw/rail_transport.csv` | `Rail_Transport.csv` |

## Key columns

`node_id.csv`: `project_id, size_tier, node_id`. `size_tier` 0 is the project node; 1–5
are SZ1–SZ5.

`size_level_node.csv` (one row per size-tier node and month):

| Column | Meaning |
|---|---|
| `node_id`, `project_id`, `timestep`, `size_tier` | keys |
| `rent_per_sqft` | observed price per sq ft (**also used for sale**; the name is generic) |
| `rent_per_sqft_imp` | imputed price per sq ft (always present) |
| `y_mask` | 1 when the price was observed that month |
| `size_tier_1` … `size_tier_5` | one-hot size tier |

`project_level_node.csv` (one row per project and month) holds about 80 (rent) or
100 (sale) numeric features:

| Family | Examples |
|---|---|
| Physical | median unit area, age at transaction, tenure class and remaining years, development size |
| Location | postal district; distance to nearest MRT, bus stop, supermarket, park, clinic, bank, ATM, school |
| Amenity density | counts of bus stops, supermarkets, parks, clinics, banks, ATMs within 500 m |
| Calendar | month of year, post-COVID indicator |
| Macroeconomic | URA rental and price index growth, SORA 3-month, GVA growth, CPI, unemployment, 10-year bond yield, SGD/USD, HDB resale index |
| Market price history | global, district and project rolling medians (1, 3, 12 months) and trends, with 12- and 24-month lagged copies |

Identifier columns (`project_id_feat`, `timestep_feat`, …) are in the tables but are
**excluded** from model inputs, as is the observed price.

## Splits

All models use the same split on the **feature month** (the month whose data is used
to predict `h` months ahead):

| Part | Feature months | Used for |
|---|---|---|
| Training | before Jan 2018 | fitting weights |
| Validation | Jan 2018 – Dec 2019 (last 24 months before the test) | early stopping, choosing hyperparameters, relations and temporal module |
| Test | Jan 2020 onward | reported once |

Input windows that cross the January 2020 boundary contain test-period targets;
those positions are masked out of training.

## Data quality

- **Imputed history.** `rent_per_sqft_imp` was produced upstream (a SAITS-style
  imputer). Build metadata of an earlier release describes it as having no time
  cutoff, so it may have been fitted on the full period. It is used as an input and to
  build the price-peer relation, so inputs may be slightly optimistic.
- **Normalisation statistics** (written beside each node table) are computed over the
  whole table, including the test period.
- **Missing coordinates.** In Full SG, one rent project (KANOPY @ 77) and 99 sale
  projects could not be geocoded. They keep their size-project and price-peer links
  but get no location- or MRT-based links, and the demo cannot draw them.
- **Age.** `condo_age_at_transaction` in `project_info.csv` is the age at one
  unspecified transaction; it is not a reliable project age.
