# Time Series Forecasting

## Business Problem

Forecast future daily temperatures for Washington using historical temperature data.

## Dataset

Daily Climate Temperature Dataset.

## Feature Engineering

- Lag 1
- Lag 7
- Lag 14
- Rolling Mean 7
- Rolling Mean 14

## Models

- Naive Baseline
- Random Forest Regressor

## Metrics

- MAE
- RMSE
- MAPE

## Results

The Random Forest model improved forecasting compared to the naive baseline by leveraging lag and rolling statistical features.

## Limitations

- External factors are not considered.
- Holidays and unusual weather events are not modeled.