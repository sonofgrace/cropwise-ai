from src.risk.yield_risk import calculate_climate_risk_score


def test_low_risk_climate_summary():
    climate_summary = {
        "total_rainfall": 1200,
        "mean_temperature": 28,
        "max_temperature": 34,
        "dry_days": 100,
        "heavy_rain_days": 10,
    }

    result = calculate_climate_risk_score(climate_summary)

    assert result["risk_score"] == 0
    assert result["risk_level"] == "Low"
    assert result["warnings"] == []


def test_moderate_risk_climate_summary():
    climate_summary = {
        "total_rainfall": 300,
        "mean_temperature": 28,
        "max_temperature": 34,
        "dry_days": 200,
        "heavy_rain_days": 10,
    }

    result = calculate_climate_risk_score(climate_summary)

    assert result["risk_score"] == 3
    assert result["risk_level"] == "Moderate"
    assert len(result["warnings"]) == 2


def test_high_risk_climate_summary():
    climate_summary = {
        "total_rainfall": 3000,
        "mean_temperature": 34,
        "max_temperature": 42,
        "dry_days": 210,
        "heavy_rain_days": 40,
    }

    result = calculate_climate_risk_score(climate_summary)

    assert result["risk_score"] == 8
    assert result["risk_level"] == "High"
    assert len(result["warnings"]) == 5


def test_climate_risk_result_has_expected_keys():
    climate_summary = {
        "total_rainfall": 1200,
        "mean_temperature": 28,
        "max_temperature": 34,
        "dry_days": 100,
        "heavy_rain_days": 10,
    }

    result = calculate_climate_risk_score(climate_summary)

    assert set(result.keys()) == {"risk_score", "risk_level", "warnings"}