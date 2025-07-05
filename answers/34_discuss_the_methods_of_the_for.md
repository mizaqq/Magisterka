# Question 34: Discuss the methods of the forecast quality assessment.

**Status:** ✅ Completed | **Words:** 184 | **Generated:** 2025-07-04T00:18:00Z

---

## Answer

Forecast accuracy is judged by comparing predicted values with realised outcomes through error-based metrics and statistical tests. Scale-dependent measures such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) quantify average deviation; RMSE penalises large misses more strongly. Percentage metrics like Mean Absolute Percentage Error (MAPE) and symmetric MAPE allow cross-series comparison when zeros are rare. For probabilistic forecasts, the Continuous Ranked Probability Score and log-likelihood evaluate full predictive distributions, while Prediction Interval Coverage Probability checks calibration. Relative metrics—Theil's U or scaled MAE—benchmark against naïve or seasonal random-walk models. Out-of-sample validation should employ rolling or expanding windows to mimic real-time learning, and the Diebold–Mariano or Giacomini–White tests determine whether observed error differences between competing models are statistically significant. Visual diagnostics—error histograms, fan charts, and QQ-plots—reveal bias or heteroscedasticity that summary numbers can hide. Combining multiple metrics provides a balanced view of forecast quality across magnitude, direction, and risk dimensions.

---
