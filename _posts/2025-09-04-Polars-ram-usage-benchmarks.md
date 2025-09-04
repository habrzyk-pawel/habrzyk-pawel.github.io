---
layout: default
title:  "Polars ram usage benchmarks"
date:   2025-06-01 09:00:00 +02:00
---

This entry is meant as an mini extension of [this beautiful blog post](https://medium.com/dev-jam/wrestling-the-bear-benchmarking-execution-modes-of-polars-8b2626efd643)

What we will do is we will try to gauge the differences between eagar, lazy and streaming in terms of ram usage.
I curently face multiple issues related to OOM, anything that mentions out of core computation is of interest to me.

## Out of core
This is a mode of computation in which a computation engine offloads whatever it has cached for future joins to a dediated file. If we are not usign something now but instead have it saved up for the future, we should store it on disk.
This way, we delay the moment where we absolutelly need a cluster to make certain aggregations. If You are facing a OOM, a good first step is to parse throught docs of the tool You are using for out of core entry (most frameworks disable it for speed).

With that out of the way, lets try to figure out the impact it might have on consumption

## Polars
We will run 3 agggreagations, each with `_eagar=True`, `_eagar=False`, `streaming=True`
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
##### _eagar=True
<img width="1438" height="1128" alt="image" src="https://github.com/user-attachments/assets/3a3803f9-966e-456f-aa00-268fda214142" />

(seconds later container crushed due to OOM)

##### _eagar=False
<img width="1426" height="1121" alt="image" src="https://github.com/user-attachments/assets/46a52520-9206-47f3-bff3-5cd590ab6e11" />

##### streaming=True
<img width="1429" height="1117" alt="image" src="https://github.com/user-attachments/assets/b00eacda-3d68-4baf-8257-c7979514fd6d" />


### Script 2
```python
import polars as pl

q_taxi = (pl.scan_csv("taxi.csv").sort("total_amount"))

df = q_taxi.collect(streaming=True)
df = q_taxi.collect(_eager=False)
df = q_taxi.collect(_eager=True)
```
#### Results
##### _eagar=True
OOM thrown
<img width="1429" height="1129" alt="image" src="https://github.com/user-attachments/assets/3b752baf-2966-474f-a4bc-6ff1016e6150" />

##### _eagar=False

OOM thrown
<img width="1433" height="1112" alt="image" src="https://github.com/user-attachments/assets/6055d2e3-f63e-4824-8c5e-1533bc430dd0" />


##### streaming=True

OOM thrown
<img width="1436" height="1141" alt="image" src="https://github.com/user-attachments/assets/057ec80f-929f-47fb-85a2-eab35d32e7f8" />

### Script 3
```python
import polars as pl

q_taxi = (pl.scan_csv("taxi.csv").sort("total_amount"))

df = q_taxi.collect(streaming=True)
df = q_taxi.collect(_eager=False)
df = q_taxi.collect(_eager=True)
```
#### Results
##### _eagar=True
OOM thrown
<img width="1429" height="1129" alt="image" src="https://github.com/user-attachments/assets/3b752baf-2966-474f-a4bc-6ff1016e6150" />

##### _eagar=False

OOM thrown
<img width="1433" height="1112" alt="image" src="https://github.com/user-attachments/assets/6055d2e3-f63e-4824-8c5e-1533bc430dd0" />


##### streaming=True

OOM thrown
<img width="1436" height="1141" alt="image" src="https://github.com/user-attachments/assets/057ec80f-929f-47fb-85a2-eab35d32e7f8" />



