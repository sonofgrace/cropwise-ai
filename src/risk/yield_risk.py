from __future__ import annotations


def calculate_climate_risk_score(climate_summary: dict) -> dict:
    """
    Calculate a simple rule-based climate risk score.

    This is not a trained yield model. It is an interpretable first version
    of a climate-risk layer.
    """
    risk_points = 0
    warnings = []

    total_rainfall = climate_summary.get("total_rainfall", 0)
    mean_temperature = climate_summary.get("mean_temperature", 0)
    max_temperature = climate_summary.get("max_temperature", 0)
    dry_days = climate_summary.get("dry_days", 0)
    heavy_rain_days = climate_summary.get("heavy_rain_days", 0)

    if total_rainfall < 500:
        risk_points += 2
        warnings.append("Low annual rainfall may increase drought risk.")

    if total_rainfall > 2500:
        risk_points += 2
        warnings.append("Very high rainfall may increase flooding or disease risk.")

    if mean_temperature > 32:
        risk_points += 2
        warnings.append("High average temperature may increase heat stress risk.")

    if max_temperature > 40:
        risk_points += 2
        warnings.append("Extreme maximum temperature may affect crop performance.")

    if dry_days > 180:
        risk_points += 1
        warnings.append("High number of dry days may require irrigation planning.")

    if heavy_rain_days > 30:
        risk_points += 1
        warnings.append(
            "Frequent heavy rainfall may increase erosion or waterlogging risk."
        )

    if risk_points <= 2:
        risk_level = "Low"
    elif risk_points <= 5:
        risk_level = "Moderate"
    else:
        risk_level = "High"

    return {
        "risk_score": risk_points,
        "risk_level": risk_level,
        "warnings": warnings,
    }