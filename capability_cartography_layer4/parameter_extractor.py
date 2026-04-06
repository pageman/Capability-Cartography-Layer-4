from typing import Dict, List


def estimate_m(case_spec: Dict) -> Dict[str, object]:
    kind = case_spec["kind"]
    if kind in {"periodic", "non_periodic"}:
        value = 17
        method = "frozen_residue_class_count"
        uncertainty = "low"
    else:
        value = 24
        method = "synthetic_support_partition_count"
        uncertainty = "medium"
    return {"value": value, "method": method, "uncertainty": uncertainty}


def estimate_r(case_spec: Dict) -> Dict[str, object]:
    kind = case_spec["kind"]
    value = 1.0
    method = "effective_samples_per_environment_proxy"
    uncertainty = "medium" if kind != "periodic" else "low"
    return {"value": value, "method": method, "uncertainty": uncertainty}


def estimate_d(case_spec: Dict) -> Dict[str, object]:
    kind = case_spec["kind"]
    if kind == "periodic":
        value = 64
    elif kind == "non_periodic":
        value = 8
    else:
        value = 48
    return {
        "value": value,
        "method": "frozen_representational_difficulty_proxy",
        "uncertainty": "high",
    }


def estimate_s_star(case_spec: Dict) -> Dict[str, object]:
    kind = case_spec["kind"]
    if kind == "periodic":
        value = 4
    elif kind == "non_periodic":
        value = 2
    else:
        value = 3
    return {
        "value": value,
        "method": "dominant_active_factor_proxy",
        "uncertainty": "high",
    }


def extract_case_parameters(case_spec: Dict) -> Dict[str, Dict[str, object]]:
    return {
        "m": estimate_m(case_spec),
        "r": estimate_r(case_spec),
        "d": estimate_d(case_spec),
        "s_star": estimate_s_star(case_spec),
    }


def flatten_parameters(extracted: Dict[str, Dict[str, object]]) -> Dict[str, float]:
    return {key: float(value["value"]) for key, value in extracted.items()}
