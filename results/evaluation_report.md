# 📊 Model Evaluation & Metrics Report

## Benchmark Metrics Across Forecast Horizons (t+1 .. t+4)

```text
horizon     wape       mae      rmse       r2  zero_count_residual
    t+1 0.380824 47.235578 55.265832 0.135342           113.717829
    t+2 0.382558 47.447672 55.502803 0.128042           113.369099
    t+3 0.381421 47.302703 55.385123 0.131825           117.647123
    t+4 0.381519 47.319641 55.406075 0.131324           114.441344
```

## Top Feature Importance Rankings

```text
               feature  xgb_importance  lgb_importance  avg_importance
              lag_168h        0.033742        0.046924        0.040333
             windspeed        0.029529        0.048662        0.039095
               lag_48h        0.035408        0.042405        0.038906
humidity_roll_mean_24h        0.032169        0.042753        0.037461
               lag_24h        0.032102        0.042753        0.037427
                 count        0.027303        0.045881        0.036592
     temp_roll_mean_3h        0.035381        0.037539        0.036460
                lag_2h        0.031366        0.041363        0.036364
              humidity        0.030761        0.041710        0.036235
 humidity_roll_mean_3h        0.031581        0.038929        0.035255
     temp_roll_std_24h        0.034998        0.035454        0.035226
     Wind Speed (km/h)        0.030784        0.036496        0.033640
```
