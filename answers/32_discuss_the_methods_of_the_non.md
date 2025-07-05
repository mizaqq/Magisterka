# Question 32: Discuss the methods of the nonstationary time series forecasting.

**Status:** ✅ Completed | **Words:** 170 | **Generated:** 2025-07-04T00:16:00Z

---

## Answer

Forecasting non-stationary series begins by identifying the source of non-stationarity—deterministic trend, seasonality, or stochastic unit root—and applying a method tailored to each. Integrated ARIMA models difference the data until stationarity, then forecast in the differenced space before re-accumulating; seasonal ARIMA adds seasonal differencing and harmonic terms. When multiple integrated variables share equilibrium, Vector Error-Correction Models combine cointegration with short-run dynamics. For structural breaks or evolving parameters, state-space models with Kalman filtering or time-varying coefficients adapt recursively, while exponential smoothing (Holt or Holt-Winters) handles local level and trend without explicit differencing. Machine-learning approaches—gradient boosting, recurrent neural networks, transformers—implicitly learn non-stationary patterns given enough data, often after windowing or detrending. Finally, combination forecasts average outputs from differing models to hedge misspecification risk. Model selection relies on out-of-sample tests like rolling-window RMSE or Diebold–Mariano statistics to ensure that the chosen approach genuinely captures non-stationary behaviour.

---
