---
layout: default
title:  "Polars ram usage benchmarks"
date:   2025-09-04 09:00:00 +02:00
---

This entry is meant as an mini extension of [this beautiful blog post](https://medium.com/dev-jam/wrestling-the-bear-benchmarking-execution-modes-of-polars-8b2626efd643)

What we will do is we will try to gauge the differences between eagar, lazy and streaming in terms of ram usage.
I curently face multiple issues related to OOM, anything that mentions out of core computation is of interest to me.
We will pit polars against DuckDB for comparison.

## Out of core
This mode of computation allows the engine to offload cached data for future joins onto a dedicated file. Instead of keeping everything in memory, data that is not immediately needed is stored on disk for later use. This approach helps defer the point at which a cluster becomes necessary for performing certain aggregations.

If you encounter an out-of-memory (OOM) issue, a good initial step is to review the documentation of the tool you are using to enable out-of-core execution (noting that many frameworks disable this feature by default in favor of speed).

With that context established, we can now examine the potential impact this approach may have on resource consumption.

## Dataset 

We will use faker to generate a simulated taxi dataset.

```python
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

  
write_csv_approx_Ngb("taxi_15gb.csv", target_gb=8, batch_size=20000)

```


## Tests
We will run 3 agggreagations, each with `_eagar=True`, `_eagar=False`, `streaming=True` and in DuckDB
(streaming is a special case of `_eagar=False`)

### Script 1
```python
import polars as pl

q_taxi = (
    pl.scan_csv("taxi.csv")
    .filter(
        (pl.col("passenger_count") >= 1) & (pl.col("trip_distance") > 0.5)
    )
    .group_by("payment_type")
    .agg(
        trips=pl.len(),
        mean_fare=pl.col("total_amount").mean(),
        mean_dist=pl.col("trip_distance").mean(),
    )
)

df = q_taxi.collect(streaming=True)
df = q_taxi.collect(_eager=False)
df = q_taxi.collect(_eager=True)
```
#### Results
* note that in the future _eagar=True -> optimizations=True and streming=True -> engine="streaming"
##### _eager=True
(OOM was thrown)


<img style="max-width:100%; height:auto;" alt="image" src="https://github.com/user-attachments/assets/3a3803f9-966e-456f-aa00-268fda214142" />

##### _eager=False
<img style="max-width:100%; height:auto;" alt="image" src="https://github.com/user-attachments/assets/46a52520-9206-47f3-bff3-5cd590ab6e11" />

##### streaming=True
<img style="max-width:100%; height:auto;" alt="image" src="https://github.com/user-attachments/assets/b00eacda-3d68-4baf-8257-c7979514fd6d" />

##### DuckDB
```python
import duckdb

con = duckdb.connect()

q = """
SELECT
  payment_type,
  COUNT(*) AS trips,
  AVG(total_amount) AS mean_fare,
  AVG(trip_distance) AS mean_dist
FROM 'taxi.csv'
WHERE passenger_count >= 1
  AND trip_distance > 0.5
GROUP BY payment_type
"""

print("streaming")
df = con.execute(q).df()

```

<img style="max-width:100%; height:auto;" alt="image" src="https://github.com/user-attachments/assets/112fa1ba-60be-41c5-98bd-479e1f2c7da7" />


### Script 2 
(script 1 with exploded group count)
```python
import polars as pl

q_taxi = (
    pl.scan_csv("taxi.csv")\
  .with_columns((pl.col("total_amount")*100).cast(pl.Int64).alias("amt_cents"))\
  .group_by(["payment_type","amt_cents"])\
  .agg(pl.len())
)

df = q_taxi.collect(streaming=True)
df = q_taxi.collect(_eager=False)
df = q_taxi.collect(_eager=True)
```
#### Results
##### _eager=True
(OOM was thrown)

<img style="max-width:100%; height:auto;" alt="image" src="https://github.com/user-attachments/assets/7b0b6c89-28a1-4ce5-b213-b3fda15a7b0e" />

##### _eager=False
<img style="max-width:100%; height:auto;" alt="image" src="https://github.com/user-attachments/assets/522ff92f-8b31-470f-bae9-26ff79cdd78e" />


##### streaming=True

<img style="max-width:100%; height:auto;" alt="image" src="https://github.com/user-attachments/assets/e527f341-0025-453b-b74f-bf00798cdd78" />

##### DuckDB

```python
import duckdb

con = duckdb.connect()

q = """
SELECT
  payment_type,
  CAST(total_amount * 100 AS BIGINT) AS amt_cents,
  COUNT(*) AS len
FROM 'taxi.csv'
GROUP BY payment_type, amt_cents
"""

print("streaming")
df = con.execute(q).df()
```

<img style="max-width:100%; height:auto;" alt="image" src="https://github.com/user-attachments/assets/17486eb2-2390-4145-a721-774a438bb4e3" />


### Script 3
```python
import polars as pl

q_taxi = (pl.scan_csv("taxi.csv").sort("total_amount"))

df = q_taxi.collect(streaming=True)
df = q_taxi.collect(_eager=False)
df = q_taxi.collect(_eager=True)
```
#### Results
##### _eager=True
(OOM was thrown)

<img style="max-width:100%; height:auto;" alt="image" src="https://github.com/user-attachments/assets/3b752baf-2966-474f-a4bc-6ff1016e6150" />

##### _eager=False

(OOM was thrown)

<img style="max-width:100%; height:auto;" alt="image" src="https://github.com/user-attachments/assets/6055d2e3-f63e-4824-8c5e-1533bc430dd0" />


##### streaming=True

(OOM was thrown)

<img style="max-width:100%; height:auto;" alt="image" src="https://github.com/user-attachments/assets/057ec80f-929f-47fb-85a2-eab35d32e7f8" />

#### DuckDB
NO OOM !!!

```python
import duckdb

con = duckdb.connect()

q = """
SELECT *
FROM 'taxi.csv'
ORDER BY total_amount
"""

df = con.execute(q).df()
```

<img style="max-width:100%; height:auto;" alt="image" src="https://github.com/user-attachments/assets/92046dcd-63ab-4b81-b2db-426e86fa108c" />


## Conclusion 
The difference can be stark - script 1&2 demonstrate that certain aggregations can be practically done without ram usage. On the other hand, script 3 shows that this is not a silver bullet. Interestingly, DuckDB survived workload 3, and I don't know what to do with it yet.

Next, we will investigate reason for worse performance of polars in this particular benchmark. We will optimze both polars and duckdb for working memory performance. After that, we will evaluate XGBoost out-of-core features to see how far we can push it on limited hardware. 



