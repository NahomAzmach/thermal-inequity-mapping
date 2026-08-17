# Nighttime thermal test (MODIS)

## Adjusted informal effect by time-of-day (per-pixel model)

Negative = informal cooler; positive = informal hotter. `effect` = class coefficient in `LST ~ class + NDVI + elevation`.

| Year | Source | Effect (°C) | 95% CI | Raw ΔLST |
|---|---|---|---|---|
| 2017 | Landsat day | -0.445 | [-0.481, -0.409] | -0.77 |
| 2017 | MODIS day | -0.446 | [-0.471, -0.421] | -0.77 |
| 2017 | MODIS night | +0.298 | [+0.272, +0.324] | -0.44 |
| 2024 | Landsat day | -0.380 | [-0.415, -0.345] | -0.87 |
| 2024 | MODIS day | -0.205 | [-0.225, -0.186] | -0.74 |
| 2024 | MODIS night | -0.075 | [-0.112, -0.038] | -0.91 |

## Purity-restricted 1 km-cell test (cleaner contrast)

Only MODIS cells that are >=70% one class; compares NIGHT LST.

| Year | informal cells | other cells | raw night Δ (°C) | adjusted Δ (°C) | 95% CI | p |
|---|---|---|---|---|---|---|
| 2017 | 97 | 947 | -0.08 | +1.131 | [+0.803, +1.460] | 2.38e-11 |
| 2024 | 339 | 631 | -2.77 | +0.027 | [-0.271, +0.325] | 8.60e-01 |

## Read
- Daytime Landsat effect: -0.405 / -0.379 °C (cooler).
- MODIS night effect: +0.298 / -0.075 °C.
- Mixed/ambiguous across years — no clean night reversal.
