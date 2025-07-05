# Question 20: Discuss the methods of the non-stationary time series analysis.

**Status:** ✅ Completed | **Words:** 166 | **Generated:** 2025-07-04T00:00:00Z

---

## Answer

Non-stationary time series are sequences whose statistical properties change over time, so traditional stationary models mislead inference. Econometric analysis therefore starts by detecting non-stationarity with unit-root or structural-break tests (Augmented Dickey–Fuller, KPSS, Zivot–Andrews). If a unit root is present, the series is differenced or log-differenced to achieve stationarity, leading to ARIMA or seasonal ARIMA forecasts; deterministic trends may instead be removed by regression detrending. When several I(1) variables share a long-run equilibrium, cointegration techniques such as Johansen's test allow Vector Error-Correction Models that blend short-run dynamics with long-run constraints. For structural breaks or evolving volatility, state-space formulations and Kalman filters, GARCH family models, or rolling-window estimators adapt parameters over time. Alternatively, non-linear methods like regime-switching and wavelets capture shifts between persistent states and high-frequency shocks. Robust analysis always couples transformation diagnostics with out-of-sample validation to ensure that the model's residuals satisfy the stationarity assumption. In sum, non-stationary analysis revolves around identifying the source of instability, transforming or co–modelling it, and then applying appropriately adjusted forecasting tools.

---
