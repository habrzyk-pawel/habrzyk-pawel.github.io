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




