from typing import Dict, List


def early_mechanism_summary(checkpoints: List[Dict[str, float]]) -> Dict[str, float]:
    if len(checkpoints) < 3:
        return {
            "early_fourier_gain": 0.0,
            "early_completeness_gain": 0.0,
            "early_rmse_drop": 0.0,
        }
    start = checkpoints[0]
    early = checkpoints[min(2, len(checkpoints) - 1)]
    return {
        "early_fourier_gain": round(early.get("fourier_signal", 0.0) - start.get("fourier_signal", 0.0), 4),
        "early_completeness_gain": round(
            early.get("circuit_completeness", 0.0) - start.get("circuit_completeness", 0.0), 4
        ),
        "early_rmse_drop": round(start.get("score_rmse", 0.0) - early.get("score_rmse", 0.0), 4),
    }


def feedback_adjust_forecast(forecast: Dict[str, object], checkpoints: List[Dict[str, float]]) -> Dict[str, object]:
    summary = early_mechanism_summary(checkpoints)
    adjusted = dict(forecast)
    adjusted["feedback_summary"] = summary
    adjusted["feedback_status"] = "unchanged"

    forecast_type = str(forecast.get("forecast_type"))
    confidence = float(forecast.get("confidence", 0.0))

    if forecast_type == "step-function":
        if summary["early_fourier_gain"] > 0.3 and summary["early_rmse_drop"] > 0.2:
            adjusted["confidence"] = round(min(0.99, confidence + 0.03), 4)
            adjusted["feedback_status"] = "confirmed_by_mechanism"
        else:
            adjusted["confidence"] = round(max(0.5, confidence - 0.15), 4)
            adjusted["feedback_status"] = "forecast_at_risk"
    elif forecast_type in {"emergent", "step-function"}:
        if summary["early_completeness_gain"] < 0.05 and summary["early_rmse_drop"] < 0.05:
            adjusted["confidence"] = round(max(0.5, confidence - 0.2), 4)
            adjusted["feedback_status"] = "forecast_at_risk"
    else:
        if summary["early_fourier_gain"] == 0.0 and summary["early_rmse_drop"] > 0.05:
            adjusted["confidence"] = round(min(0.99, confidence + 0.02), 4)
            adjusted["feedback_status"] = "consistent_with_smooth_progress"

    return adjusted
