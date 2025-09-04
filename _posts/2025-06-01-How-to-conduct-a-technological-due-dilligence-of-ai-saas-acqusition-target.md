---
layout: default
title:  "What is MLOps"
date:   2025-06-01 09:00:00 +02:00
---

This entry is meant as an mini extension of [this beautiful blog post]()

What we will do is we will try to gauge the differences between eagar, lazy and streaming in terms of ram usage.
I curently face multiple issues related to OOM, anything that mentions out of core computation is of interest to me.

## Out of core
This is a mode of computation in which a computation engine offloads whatever it has cached for future joins to a dediated file. If we are not usign something now but instead have it saved up for the future, we should store it on disk.
This way, we delay the moment where we absolutelly need a cluster to make certain aggregations. If You are facing a OOM, a good first step is to parse throught docs of the tool You are using for out of core entry (most frameworks disable it for speed).

With that out of the way, lets try to figure out the impact it might have on consumption

## Polars
We will run 3 agggreagations, each with `_eagar=True`, `_eagar=False`, `streaming=True`
(streaming is a special case of `_eagar=False`)


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
### Results
#### _eagar=True
<img width="708" height="552" alt="image" src="https://github.com/user-attachments/assets/278abd81-7b77-40fc-bc74-94966052229e" />

#### _eagar=False
<img width="700" height="545" alt="image" src="https://github.com/user-attachments/assets/1084427c-4b99-4ed9-bf44-2c72ec3b1e7f" />

#### streaming=True
<img width="697" height="544" alt="image" src="https://github.com/user-attachments/assets/1431ccec-3b59-40c6-bf9d-07bf5a0fe45f" />

