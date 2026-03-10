# Real-World Datasets

This directory contains three real-world datasets and their processing scripts.
Each processor reads its raw data, encodes records as integers in `{0, …, U}`,
and outputs a CSV file in the unified `UserLevelDataset` format consumed by
`run_experiment.py`.

## Common Output Format

All processors output CSV files with the same structure:

```
# n=1000,M=64,U=10000000
# source=aol,raw_file=user-ct-test-collection-01.txt,...
82741,9283104,28193,...
992847
...
```

- **Line 1** (required): `# n=<users>,M=<max_records>,U=<domain_upper_bound>`
- **Line 2** (metadata): `# key=value,key=value,...`
- **Data lines**: one per user, comma-separated integers; blank line = user with 0 records

All values satisfy `0 ≤ value ≤ U`. The header is parsed by
`UserLevelDataset.load_csv()` and used to initialize protocol parameters.

---

## Full-Dataset Characteristics Under Current Preprocessing

The following statistics are computed from the raw files in this directory
under the current extraction and encoding rules, **before** any `--n` user
subsampling and **before** any `--M` per-user clipping. They therefore
describe the complete processed datasets, not a particular experiment output.

| Dataset | Encoding / retained record definition | Users with >= 1 retained record | Retained records | Min per user | Max per user | Avg per user |
|---------|---------------------------------------|----------------------------------|------------------|--------------|--------------|--------------|
| AOL | Domain extracted from `ClickURL` or domain-like `Query` | 58,907 | 2,169,569 | 1 | 2,889 | 36.83 |
| MovieLens | `movie_rating` | 200,948 | 32,000,204 | 20 | 33,332 | 159.25 |
| MovieLens | `movie_id` | 200,948 | 32,000,204 | 20 | 33,332 | 159.25 |
| Netflix | `movie_rating_date` | 480,189 | 100,480,507 | 1 | 17,653 | 209.25 |
| Netflix | `movie_rating` | 480,189 | 100,480,507 | 1 | 17,653 | 209.25 |

Notes:
- AOL discards rows for which no valid domain can be extracted, so the number of users after preprocessing is much smaller than the raw number of distinct `AnonID`s.
- MovieLens `movie_rating` and `movie_id` have the same user and record counts because `ratings.csv` contains only valid half-star ratings; the mode changes only the integer encoding.
- Netflix `movie_rating_date` and `movie_rating` have the same user and record counts because the local `training_set/` contains valid ratings and valid ISO-format dates throughout; the mode changes only the integer encoding.

---

## 1. AOL Search Log (`aol/`)

| Property | Value |
|----------|-------|
| Raw file | `user-ct-test-collection-01.txt` |
| Format | Tab-separated: `AnonID  Query  QueryTime  ItemRank  ClickURL` |
| Raw scale | ~3.5M rows, ~650K unique users |
| Script | `process_aol.py` |

### How records are extracted

Each row is a search event. A "record" is extracted from:

1. **ClickURL** (priority) — the actual clicked URL
2. **Query** — if it looks like a domain name (contains `.`, no spaces, matches
   known TLDs like `.com`, `.org`, `.net`, etc.)
3. Otherwise the row is skipped

### Encoding: URL domain → integer

```
1.  Extract domain from URL (strip http(s)://, www.)
2.  Take the first 3 characters of the domain (pad with 'a' if shorter)
3.  Convert each character to 8-bit binary → concatenate → 24-bit integer
4.  value = raw_integer % (U + 1)
```

- **Max raw value**: 2^24 - 1 = 16,777,215 ≈ 1.68 × 10^7
- **Recommended U**: up to 10^7

### Usage

```bash
cd experiment/dataset/real_data/aol
python process_aol.py --n 5000 --M 32 --U 1048576
# Output: data/aol_n5000_M32_U1048576.csv
```

---

## 2. MovieLens 32M (`ml-32m/`)

| Property | Value |
|----------|-------|
| Raw file | `ratings.csv` |
| Format | CSV: `userId,movieId,rating,timestamp` |
| Raw scale | ~32M rows, ~200K users, ~87K movies |
| Script | `process_movielens.py` |

Ratings use half-star increments: 0.5, 1.0, 1.5, …, 5.0 (10 levels).
Data is sorted by `userId`.

### Encoding modes

#### `movie_rating` (default)

Combines movie identity and rating into a single integer:

```
half_star_index = int(rating × 2) - 1       ∈ {0, 1, …, 9}
raw = (movieId - 1) × 10 + half_star_index
value = raw % (U + 1)
```

- **Max raw value**: (292,757 - 1) × 10 + 9 = 2,927,569 ≈ 2.93 × 10^6
- **Recommended U**: 10^6 – 10^7

Each `(movie, rating)` pair maps to a distinct domain element.

#### `movie_id`

Ignores the rating; only uses movie identity:

```
raw = movieId - 1
value = raw % (U + 1)
```

- **Max raw value**: 292,756
- **Recommended U**: up to 3 × 10^5

### Usage

```bash
cd experiment/dataset/real_data/ml-32m

# Default (movie_rating)
python process_movielens.py --n 5000 --M 64 --U 10000000
# Output: data/ml32m_n5000_M64_U10000000.csv

# Movie-ID only
python process_movielens.py --n 5000 --M 64 --U 300000 --mode movie_id
```

---

## 3. Netflix Prize (`Netf/`)

| Property | Value |
|----------|-------|
| Raw data | `training_set/` directory (17,770 files, one per movie) |
| File format | First line: `MovieID:`, then `CustomerID,Rating,Date` per line |
| Raw scale | ~100M ratings, ~480K users, 17,770 movies |
| Ratings | Integer 1–5 (5 levels) |
| Date range | October 1998 – December 2005 |
| Script | `process_netflix.py` |

Unlike AOL and MovieLens (single flat file), Netflix stores data
**per movie** — every `mv_NNNNNNN.txt` file contains all ratings for that movie.
The processor reads all 17,770 files and aggregates by user.

### Encoding modes

#### `movie_rating_date` (default)

Combines movie identity, rating, and date to produce a large-domain integer:

```
day_offset = days since 1998-10-01        ∈ {0, …, ~2648}
raw = (movieId - 1) × 100000 + day_offset × 5 + (rating - 1)
value = raw % (U + 1)
```

- **Max raw value**: 17,769 × 100,000 + 2,648 × 5 + 4 = 1,776,913,244 ≈ **1.78 × 10^9**
- **Recommended U**: **10^7 – 10^9**

The date dimension ensures that the same (movie, rating) pair on different days
maps to different values, providing millions of distinct domain elements and
natural coverage across large domains.

#### `movie_rating`

Simpler encoding that ignores the date:

```
raw = (movieId - 1) × 5 + (rating - 1)
value = raw % (U + 1)
```

- **Max raw value**: 17,769 × 5 + 4 = 88,849
- **Recommended U**: up to 10^5

### Usage

```bash
cd experiment/dataset/real_data/Netf

# Large domain (default movie_rating_date)
python process_netflix.py --n 5000 --M 64 --U 1000000000
# Output: data/netflix_n5000_M64_U1000000000.csv

# Small domain
python process_netflix.py --n 5000 --M 64 --U 100000 --mode movie_rating
```

**Note**: Reading 17,770 movie files takes approximately 2–3 minutes.

---

## Summary: Encoding Comparison

| Dataset | Mode | Formula | Max raw value | Recommended U |
|---------|------|---------|---------------|---------------|
| AOL | — | 3-char domain → 24-bit int | 1.68 × 10^7 | ≤ 10^7 |
| MovieLens | `movie_rating` | (movieId-1)×10 + half_star | 2.93 × 10^6 | 10^6 – 10^7 |
| MovieLens | `movie_id` | movieId - 1 | 2.93 × 10^5 | ≤ 3×10^5 |
| Netflix | `movie_rating_date` | (movieId-1)×100000 + day×5 + rating | **1.78 × 10^9** | **10^7 – 10^9** |
| Netflix | `movie_rating` | (movieId-1)×5 + rating | 8.88 × 10^4 | ≤ 10^5 |

All encodings apply `value = raw % (U + 1)` to map into `[0, U]`.

## Common CLI Arguments

All three processors share the same interface:

| Argument | Description |
|----------|-------------|
| `--n` | Number of users to include (required) |
| `--M` | Max records per user (required) |
| `--U` | Domain upper bound, values in {0, …, U} (required) |
| `--raw_data` | Path to raw data file/directory (auto-detected by default) |
| `--mode` | Encoding mode (MovieLens and Netflix only) |
| `--output` | Exact output path (overrides auto-naming) |
| `--output_dir` | Directory for auto-named output |
| `--quiet` | Suppress summary output |

## Running Experiments with Real Data

```bash
# Step 1: Process raw data
cd experiment/dataset/real_data/Netf
python process_netflix.py --n 5000 --M 64 --U 1000000000

# Step 2: Run experiment
cd experiment/static && python run_experiment.py --dataset ../dataset/real_data/Netf/data/netflix_n5000_M64_U1000000000.csv --quick_fe1 --output results_netflix.csv
```

Notes:
- `--quick_fe1` expands to common FE1 settings (4 protocols, `epsilon=1.0`, `times=50`, `trim=0.2`).
- `--output results_netflix.csv` is timestamped by default (for example, `results_netflix_20260227_170000.csv`).
