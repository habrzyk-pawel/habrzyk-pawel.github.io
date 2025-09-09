---
layout: default
title:  "Polars ram usage benchmarks - Pt 2"
date:   2025-09-08 09:00:00 +02:00
---


Pt 2 of [Polars ram usage benchmarks](https://habrzyk-pawel.github.io/2025/09/04/Polars-ram-usage-benchmarks.html)

## Intro
Upgrading hardware takes time. Training an ml model is fast or slow due to processing time - the productivity per hour is the *soft limit*. Memory on the other hand is a hard limit - if we run out of it training does not happen. This article explores what can be done to delay the need for a distributed training cluster deployment. We will compare tools. We will also tune them to use as little memory as possible *even at a cost of cpu time* 


## Recap
In the previous article we found that polars is more efficient than pandas (expected) and duckDB is more efficient than polars (unexpected). Now, we will throw more benchmarks and better profiling tools to find out if that was just an outlier or an actual phenomenon.



## Dataset

We will use faker to generate a simulated taxi dataset. This time we will use a 550mb file instead of 7.5gb

<details>
    <summary>Dataset generation code</summary>

{% highlight python %}
from faker import Faker
import random
import datetime

fake = Faker()

def generate_taxi_csv_row():
    pickup = fake.date_time_between(start_date="-30d", end_date="now")
    dropoff = pickup + datetime.timedelta(minutes=random.randint(5, 60))

    passenger_count = random.randint(1, 4)
    trip_distance = round(random.uniform(0.5, 15.0), 2)

    fare = round(trip_distance * random.uniform(2.0, 4.0), 2)
    extra = round(random.uniform(0, 5), 2)
    mta_tax = 0.5
    tip = round(fare * random.uniform(0.1, 0.3), 2)
    tolls = round(random.uniform(0, 10), 2)
    total = round(fare + extra + mta_tax + tip + tolls, 2)

    fields = [
        random.choice([1, 2]),  # vendor_id
        pickup.isoformat(sep=" "),
        dropoff.isoformat(sep=" "),
        passenger_count,
        trip_distance,
        round(random.uniform(-74.05, -73.75), 6),  # pickup_longitude
        round(random.uniform(40.63, 40.85), 6),   # pickup_latitude
        round(random.uniform(-74.05, -73.75), 6), # dropoff_longitude
        round(random.uniform(40.63, 40.85), 6),   # dropoff_latitude
        random.randint(1, 6),                     # rate_code_id
        random.choice(["Cash", "Credit", "No Charge", "Dispute"]),
        fare,
        extra,
        mta_tax,
        tip,
        tolls,
        total,
    ]

    return ",".join(map(str, fields))


def write_csv_approx_Ngb(path, target_gb=10, batch_size=100000):
    target_bytes = int(target_gb * (1024**3))
    written = 0
    with open(path, "w", buffering=1024*1024) as f:
        while written < target_bytes:
            rows = [generate_taxi_csv_row() for _ in range(batch_size)]
            block = "\n".join(rows) + "\n"
            f.write(block)
            written += len(block)
            if written // (100 * 1024 * 1024) != (written - len(block)) // (100 * 1024 * 1024):
                print(f"{written / (1024**3):.2f} GB written...")
    print(f"Done. Wrote ~{written / (1024**3):.2f} GB to {path}")

  
write_csv_approx_Ngb("taxi_550mb.csv", target_gb=0.55, batch_size=20000)
{% endhighlight %}
</details>

## Benchmark
<details>
<summary>Code</summary>
{% highlight python %}
# --- Pandas ---
FILE = "taxi_550mb.csv"
COLUMNS = [
    "vendor_id",
    "pickup_datetime",
    "dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "pickup_longitude",
    "dropoff_longitude",
    "rate_code_id",
    "payment_type",
    "extra",
    "mta_tax",
    "tip_amount",
]
 # Downcast numeric dtypes to reduce memory in pandas
PANDAS_DTYPE_MAP = {
    "vendor_id": "Int8",
    "passenger_count": "Int8",
    "rate_code_id": "Int8",
    "trip_distance": "float32",
    "pickup_longitude": "float32",
    "dropoff_longitude": "float32",
    "extra": "float32",
    "mta_tax": "float32",
    "tip_amount": "float32",
}

# Matching dtypes for Polars to avoid wide defaults
try:
    import polars as _pl
    POLARS_DTYPES = {
        "vendor_id": _pl.Int8,
        "passenger_count": _pl.Int8,
        "rate_code_id": _pl.Int8,
        "trip_distance": _pl.Float32,
        "pickup_longitude": _pl.Float32,
        "dropoff_longitude": _pl.Float32,
        "extra": _pl.Float32,
        "mta_tax": _pl.Float32,
        "tip_amount": _pl.Float32,
    }
except Exception:
    POLARS_DTYPES = None
def pandas_read_full_1():
    import pandas as pd
    pd.read_csv(FILE, usecols=COLUMNS, dtype=PANDAS_DTYPE_MAP, memory_map=True)

def polars_read_full_1():
    import polars as pl
    (
        pl.scan_csv(FILE, schema_overrides=POLARS_DTYPES)
        .select(COLUMNS)
        .collect()
    )

def duckdb_read_full_1():
    import duckdb
    # Return relation; avoid converting to a DataFrame here
    return duckdb.sql(f"SELECT * FROM {FILE!r}")

pandas_read_full_1()
polars_read_full_1()
duckdb_read_full_1()
print(1)

def pandas_read_cols_1():
    import pandas as pd
    # usecols streams only these columns from disk (memory-friendly)
    return pd.read_csv(FILE, usecols=COLUMNS, dtype=PANDAS_DTYPE_MAP, memory_map=True)
print(1.1)

def polars_read_cols_1():
    import polars as pl
    # Read just these columns; for even larger files, consider pl.scan_csv(...).select(...).collect()
    return (
        pl.scan_csv(FILE, schema_overrides=POLARS_DTYPES)
        .select(COLUMNS)
        .collect()
    )
print(1.2)

def duckdb_read_cols_1():
    import duckdb
    # Read & project only needed columns via SQL; convert to pandas DF (or drop .pl() to keep a DuckDB Relation)
    select_list = ", ".join(COLUMNS)
    # Pass file path as a parameter using the supported keyword arg
    return duckdb.sql(f"SELECT {select_list} FROM {FILE!r}")

pandas_read_cols_1()
polars_read_cols_1()
duckdb_read_cols_1()
print(2)

def pandas_read_cols_2():
    import pandas as pd
    # usecols streams only these columns from disk (memory-friendly)
    return pd.read_csv(FILE, usecols=COLUMNS, dtype=PANDAS_DTYPE_MAP, memory_map=True)
print(2.1)
def polars_read_cols_2():
    import polars as pl
    # Read just these columns; for even larger files, consider pl.scan_csv(...).select(...).collect()
    return (
        pl.scan_csv(FILE, schema_overrides=POLARS_DTYPES)
        .select(COLUMNS)
        .collect()
    )
print(2.2)
def duckdb_read_cols_2():
    import duckdb
    # Read & project only needed columns via SQL; convert to pandas DF (or drop .pl() to keep a DuckDB Relation)
    select_list = ", ".join(COLUMNS)
    # Pass file path as a parameter using the supported keyword arg
    # Keep relation; defer materialization
    return duckdb.sql(f"SELECT {select_list} FROM {FILE!r}")

pandas_read_cols_2()
polars_read_cols_2()
duckdb_read_cols_2()
print(3)

def pandas_diff_counts():
    import pandas as pd
    df = pd.read_csv(
        FILE,
        usecols=["pickup_datetime", "trip_distance"],
        dtype={"trip_distance": "float32"},
        memory_map=True,
        parse_dates=["pickup_datetime"],
    )
    df = df.sort_values("pickup_datetime")
    df["trip_distance_change"] = df["trip_distance"].diff()
    change_counts = (
        df.groupby("trip_distance_change").size()
        .sort_values(ascending=False)
        .to_frame("count")
        .reset_index()
    )
print(3.1)

def polars_diff_counts():
    import polars as pl
    (
        pl.scan_csv(FILE, schema_overrides=POLARS_DTYPES)
          .select(["pickup_datetime", "trip_distance"]) 
          .with_columns(pl.col("pickup_datetime").cast(pl.Datetime, strict=False))
          .drop_nulls(["pickup_datetime"]) 
          .sort("pickup_datetime")
          .with_columns(pl.col("trip_distance").diff().alias("trip_distance_change"))
          .drop_nulls(["trip_distance_change"]) 
          .group_by("trip_distance_change").agg(pl.len().alias("count"))
          .sort("count", descending=True)
          .collect()
    )
print(3.2)

def duckdb_diff_counts():
    import duckdb
    con = duckdb.connect()
    # Reduce memory usage: limit threads, cap memory, and enable disk spill directory
    con.execute("SET threads=1")
    con.execute("SET memory_limit='512MB'")
    con.execute("SET temp_directory='duckspill'")
    change_counts = con.sql(f"""
    WITH changes AS (
    SELECT
        trip_distance,
        trip_distance - LAG(trip_distance) OVER (ORDER BY pickup_datetime) AS trip_distance_change
    FROM {FILE!r}
    )
    SELECT trip_distance_change, COUNT(*) AS count
    FROM changes
    WHERE trip_distance_change IS NOT NULL
    GROUP BY trip_distance_change
    ORDER BY count DESC
    """)

pandas_diff_counts()
polars_diff_counts()
duckdb_diff_counts()
print(4)

def pandas_min_diff():
    import pandas as pd
    s = (
        pd.read_csv(
            FILE,
            usecols=["pickup_datetime", "trip_distance"],
            dtype={"trip_distance": "float32"},
            memory_map=True,
            parse_dates=["pickup_datetime"],
        )
          .sort_values("pickup_datetime")["trip_distance"]
          .diff()
    )
    return s.min()  # skips NaN by default
print(4.1)

def polars_min_diff():
    import polars as pl
    res = (
        pl.scan_csv(FILE, schema_overrides=POLARS_DTYPES)
          .select(["pickup_datetime", "trip_distance"]) 
          .with_columns(pl.col("pickup_datetime").cast(pl.Datetime, strict=False))
          .drop_nulls(["pickup_datetime"]) 
          .sort("pickup_datetime")
          .with_columns(pl.col("trip_distance").diff().alias("trip_distance_change"))
          .select(pl.col("trip_distance_change").min().alias("min_value"))
          .collect()
    )
    return res["min_value"][0]
print(4.2)

def duckdb_min_diff():
    import duckdb
    con = duckdb.connect()
    # Reduce memory usage
    min_val = con.sql(f"""
        WITH changes AS (
          SELECT
            trip_distance - LAG(trip_distance) OVER (ORDER BY pickup_datetime) AS trip_distance_change
          FROM {FILE!r}
        )
        SELECT MIN(trip_distance_change) AS min_value
        FROM changes
    """).fetchone()[0]
    con.close()
    return min_val

pandas_min_diff()
polars_min_diff()
duckdb_min_diff()
print(5)

def pandas_rolling_3d():
    import pandas as pd
    group_cols = ["vendor_id", "passenger_count", "payment_type"]

    df = pd.read_csv(
        FILE,
        usecols=group_cols + ["pickup_datetime", "trip_distance"],
        dtype=PANDAS_DTYPE_MAP,
        parse_dates=["pickup_datetime"],
        memory_map=True,
    )
    df = df.sort_values(group_cols + ["pickup_datetime"])

    def add_roll(g: pd.DataFrame) -> pd.DataFrame:
        s = g.set_index("pickup_datetime")["trip_distance"]
        # Ensure a proper DatetimeIndex for time-based rolling
        s.index = pd.DatetimeIndex(s.index)
        # Ensure monotonic index for time-based rolling
        s = s.sort_index()
        win = s.rolling("3D", min_periods=1)  # time-based, per group
        return g.assign(
            roll_sum_3d=win.sum().to_numpy(),
            roll_mean_3d=win.mean().to_numpy(),
            roll_count_3d=win.count().to_numpy(),
        )

    out = df.groupby(group_cols, group_keys=False).apply(add_roll)
    return out
print(5.1)

def polars_rolling_3d():
    import polars as pl
    group_cols = ["vendor_id", "passenger_count", "payment_type"]

    lf = (
        pl.scan_csv(FILE, schema_overrides=POLARS_DTYPES)
          .select(group_cols + ["pickup_datetime", "trip_distance"])  # prune early
          .with_columns(pl.col("pickup_datetime").cast(pl.Datetime, strict=False))
          .sort(group_cols + ["pickup_datetime"])  # ensure sorted within groups
          .drop_nulls(subset=["pickup_datetime"] + group_cols)  # dynamic group_by requires non-null keys/index
    )

    # Use dynamic grouping to emulate a 3-day trailing window per group.
    # Start windows at each datapoint and close on the right to include the current row.
    out = (
        lf.group_by_dynamic(
              index_column="pickup_datetime",
              every="1d",
              period="3d",
              group_by=group_cols,
              start_by="datapoint",
              closed="right",
          )
          .agg([
              pl.col("trip_distance").sum().alias("roll_sum_3d"),
              pl.col("trip_distance").mean().alias("roll_mean_3d"),
              pl.len().alias("roll_count_3d"),
          ])
          .collect()
    )
    return out
print(5.2)

def duckdb_rolling_3d():
    import duckdb
    group_cols = ["vendor_id", "passenger_count", "payment_type"]
    keys = ", ".join(group_cols)

    sql = f"""
    WITH src AS (
      SELECT *,
             CAST(pickup_datetime AS TIMESTAMP) AS ts
      FROM {FILE!r}
    )
    SELECT
      {keys},
      ts AS pickup_datetime,
      SUM(trip_distance) OVER (
        PARTITION BY {keys}
        ORDER BY ts
        RANGE BETWEEN INTERVAL '3' DAY PRECEDING AND CURRENT ROW
      ) AS roll_sum_3d,
      AVG(trip_distance) OVER (
        PARTITION BY {keys}
        ORDER BY ts
        RANGE BETWEEN INTERVAL '3' DAY PRECEDING AND CURRENT ROW
      ) AS roll_mean_3d,
      COUNT(*) OVER (
        PARTITION BY {keys}
        ORDER BY ts
        RANGE BETWEEN INTERVAL '3' DAY PRECEDING AND CURRENT ROW
      ) AS roll_count_3d
    FROM src
    ORDER BY {keys}, ts
    """
    con = duckdb.connect()
    # Reduce memory usage
    con.execute("SET threads=1")
    con.execute("SET memory_limit='512MB'")
    con.execute("SET temp_directory='duckspill'")
    out = con.sql(sql).pl()
    con.close()
    return out

pandas_rolling_3d()
polars_rolling_3d()
duckdb_rolling_3d()
print(6)

{% endhighlight %}
</details>

### Optimizations
- Pandas:
  - usecols: read only needed columns to reduce in-memory width.
  - dtype downcast: map ints to Int8 and floats to float32 where applicable.
  - memory_map: enable `memory_map=True` in `read_csv` to lower peak RSS.
  - parse_dates selectively for time-based ops to avoid unnecessary object dtype.

- Polars:
  - Lazy ingestion: switch to `scan_csv` with early `.select(...)` projection.
  - Streaming execution: use `.collect(engine="streaming")` to avoid full materialization.
  - Explicit dtypes: provide narrower numeric dtypes for selected columns.

- [DuckDB](https://duckdb.org/docs/stable/configuration/overview.html):
  - Disk spill: `SET temp_directory='duckspill'` for on-disk spilling.
  - CSV reader: `read_csv_auto(..., parallel=false)` to decrease input buffering.
  
  Theese 2 things I had to remove:
  - Lower parallelism: `SET threads=1` to reduce concurrent buffers.
  - Cap memory: `SET memory_limit='512MB'` so operators spill instead of growing RAM.
  Both cpu and disk throughput were exhausted while ram utilization was low. If need be, try to tweak them to Your needs.

### Results

<img  style="max-width:100%; height:auto;" alt="output" src="https://github.com/user-attachments/assets/0a68ba20-2afb-472f-b71a-1416ff0fea53" />





