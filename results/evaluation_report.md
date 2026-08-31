# 📊 Model Evaluation & Metrics Report

## Benchmark Metrics Across Forecast Horizons (t+1 .. t+4)

```text
horizon     wape       mae      rmse       r2  zero_count_residual
    t+1 0.377425 46.814035 54.819241 0.149260           112.301393
    t+2 0.380968 47.250475 55.278271 0.135083           111.670445
    t+3 0.375774 46.602329 54.632502 0.155260           116.141639
    t+4 0.376185 46.658112 54.710435 0.153000           111.475076
```

## Top Feature Importance Rankings

```text
               feature  xgb_importance  lgb_importance  avg_importance
              lag_168h        0.032115        0.046924        0.039519
               lag_24h        0.034450        0.042753        0.038602
             windspeed        0.028376        0.048662        0.038519
               lag_48h        0.031184        0.042405        0.036794
humidity_roll_mean_24h        0.030631        0.042753        0.036692
 humidity_roll_mean_3h        0.033528        0.038929        0.036229
     temp_roll_mean_3h        0.034566        0.037539        0.036053
                 count        0.025726        0.045881        0.035804
                lag_2h        0.029885        0.041363        0.035624
      temp_roll_std_3h        0.032812        0.036496        0.034654
     Wind Speed (km/h)        0.028726        0.036496        0.032611
    temp_roll_mean_24h        0.033161        0.031978        0.032569
```
