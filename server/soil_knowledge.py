import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_json(filename):
    path = os.path.join(BASE_DIR, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


CROP_PROFILES = _load_json("crop_profiles.json")
DISTRICT_SOIL = _load_json("district_soil.json")


def get_crop_profile(crop_name: str) -> dict:
    if not crop_name:
        return {}
    return CROP_PROFILES.get(crop_name.strip().lower(), {})


def get_district_soil(district_name: str) -> dict:
    if not district_name:
        return {}
    return DISTRICT_SOIL.get(district_name.strip().lower(), {})


def evaluate_soil_health(district: str, crop: str) -> dict:
    district_data = get_district_soil(district)
    crop_data = get_crop_profile(crop)

    if not district_data:
        return {"status": "info", "message": f"No regional soil data recorded for {district}."}

    d_n = district_data.get("avg_n", 0)
    d_p = district_data.get("avg_p", 0)
    d_k = district_data.get("avg_k", 0)
    d_ph = district_data.get("avg_ph", 7.0)

    recommendations = []

    # pH Assessment
    if d_ph < 6.0:
        recommendations.append(
            f"Soil is acidic (pH {d_ph}). Apply agricultural lime to buffer acidity.")
    elif d_ph > 7.5:
        recommendations.append(
            f"Soil is alkaline (pH {d_ph}). Consider gypsum or sulfur amendments.")

    # Target-based comparison if crop is known
    if crop_data:
        opt_n = crop_data.get("opt_n", 0)
        opt_p = crop_data.get("opt_p", 0)
        opt_k = crop_data.get("opt_k", 0)

        if d_n < opt_n * 0.8:
            recommendations.append(
                f"Low Nitrogen for {crop} (Regional: {d_n} vs Optimal: {opt_n}). Apply urea or compost.")
        if d_p < opt_p * 0.8:
            recommendations.append(
                f"Low Phosphorus for {crop} (Regional: {d_p} vs Optimal: {opt_p}). Apply DAP or rock phosphate.")
        if d_k < opt_k * 0.8:
            recommendations.append(
                f"Low Potassium for {crop} (Regional: {d_k} vs Optimal: {opt_k}). Apply MOP (Muriate of Potash).")

    return {
        "district": district,
        "regional_averages": district_data,
        "crop_profile": crop_data if crop_data else None,
        "recommendations": recommendations or ["Regional soil parameters align with standard growing ranges."]
    }
