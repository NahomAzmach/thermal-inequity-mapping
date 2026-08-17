# Thermal-inequity statistics

## Adjusted informal-settlement thermal penalty

| Year | n | Raw median ΔLST (°C) | Adjusted effect (°C) | 95% CI | p |
|---|---|---|---|---|---|
| 2017 | 97,430 | -0.76 | -0.405 | [-0.433, -0.376] | 1.16e-173 |
| 2024 | 97,462 | -0.90 | -0.379 | [-0.406, -0.352] | 3.16e-169 |

*Adjusted effect = coefficient on predicted class in `LST ~ class + NDVI + elevation`, i.e. the LST gap attributable to fabric type after removing vegetation and elevation.*

## Did the penalty widen? (pooled class×year model)

- Class effect in 2017: **-0.405 °C**
- Additional effect in 2024 (interaction): **0.025 °C** (95% CI [-0.013, 0.064], p=2.00e-01)
- positive interaction => penalty WIDENED 2017->2024; negative => narrowed; CI crossing 0 => not distinguishable
