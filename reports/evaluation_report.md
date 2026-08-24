# 📊 Model Evaluation & Metrics Report

## Benchmark Metrics Across Forecast Horizons (t+1 .. t+4)

```text
horizon     wape       mae      rmse       r2  zero_count_residual
    t+1 0.370105 45.906019 53.787042 0.180996           107.449173
    t+2 0.375945 46.627395 54.601605 0.156128           109.775898
    t+3 0.378950 46.996219 55.056348 0.142102           117.089756
    t+4 0.371359 46.059552 54.028500 0.173983           110.402462
```

## Top Feature Importance Rankings

```text
               feature  xgb_importance  lgb_importance  avg_importance
              lag_168h        0.032793        0.046924        0.039858
humidity_roll_mean_24h        0.032521        0.042753        0.037637
               lag_48h        0.031701        0.042405        0.037053
               lag_24h        0.031053        0.042753        0.036903
             windspeed        0.024161        0.048662        0.036412
                lag_2h        0.030434        0.041363        0.035898
 humidity_roll_mean_3h        0.032633        0.038929        0.035781
                 count        0.024588        0.045881        0.035235
     temp_roll_mean_3h        0.031052        0.037539        0.034295
      temp_roll_std_3h        0.031812        0.036496        0.034154
     temp_roll_std_24h        0.032684        0.035454        0.034069
      temp_roll_std_6h        0.031368        0.035454        0.033411
```
