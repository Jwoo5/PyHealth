from typing import Dict
from datetime import timedelta

from pyhealth.data import Patient
from pyhealth.tasks import register_task

@register_task("single_icu_short_term_mortality_prediction_mimic3")
def single_icu_short_term_mortality_prediction_mimic3_fn(
    patient: Patient,
    obs_size: int = 12,
    gap_size: int = 0,
    **kwargs
) -> Dict[str, int]:
    """TODO: to be written"""
    labels = {}
    
    for visit in patient:
        if (
            visit.discharge_location in [
                "Death",
                "DEAD/EXPIRED"
            ]
            and visit.encounter_time + timedelta(hours=obs_size + gap_size) <= visit.hospital_discharge_time
            and visit.hospital_discharge_time <= visit.encounter_time + timedelta(hours=obs_size + 24)
        ):
            labels[visit.visit_id] = 1
        else:
            labels[visit.visit_id] = 0

    return labels

@register_task("single_icu_long_term_mortality_prediction_mimic3")
def single_icu_long_term_mortality_prediction_mimic3_fn(
    patient: Patient,
    obs_size: int = 12,
    gap_size: int = 0,
    **kwargs
) -> Dict[str, int]:
    """TODO: to be written"""
    labels = {}
    
    for visit in patient:
        if (
            visit.discharge_location in [
                "Death",
                "DEAD/EXPIRED"
            ]
            and visit.encounter_time + timedelta(hours=obs_size + gap_size) <= visit.hospital_discharge_time
            and visit.hospital_discharge_time <= visit.encounter_time + timedelta(hours=obs_size + 336)
        ):
            labels[visit.visit_id] = 1
        else:
            labels[visit.visit_id] = 0
    
    return labels

@register_task("single_icu_short_term_mortality_prediction_mimic4")
def single_icu_short_term_mortality_prediction_mimic4_fn(
    patient: Patient,
    obs_size: int = 12,
    gap_size: int = 0,
    **kwargs
) -> Dict[str, int]:
    """TODO: to be written"""
    labels = {}
    
    for visit in patient:
        if (
            visit.discharge_location in [
                "Death",
                "DIED"
            ]
            and visit.encounter_time + timedelta(hours=obs_size + gap_size) <= visit.hospital_discharge_time
            and visit.hospital_discharge_time <= visit.encounter_time + timedelta(hours=obs_size + 24)
        ):
            labels[visit.visit_id] = 1
        else:
            labels[visit.visit_id] = 0
    
    return labels

@register_task("single_icu_long_term_mortality_prediction_mimic4")
def single_icu_long_term_mortality_prediction_mimic4_fn(
    patient: Patient,
    obs_size: int = 12,
    gap_size: int = 0,
    **kwargs
) -> Dict[str, int]:
    """TODO: to be written"""
    labels = {}
    
    for visit in patient:
        if (
            visit.discharge_location in [
                "Death",
                "DIED"
            ]
            and visit.encounter_time + timedelta(hours=obs_size + gap_size) <= visit.hospital_discharge_time
            and visit.hospital_discharge_time <= visit.encounter_time + timedelta(hours=obs_size + 336)
        ):
            labels[visit.visit_id] = 1
        else:
            labels[visit.visit_id] = 0
    
    return labels

@register_task("single_icu_short_term_mortality_prediction_eicu")
def single_icu_short_term_mortality_prediction_eicu_fn(
    patient: Patient,
    obs_size: int = 12,
    gap_size: int = 0,
    **kwargs
) -> Dict[str, int]:
    """TODO: to be written"""
    labels = {}

    for visit in patient:
        if (
            (visit.discharge_status == "Expired" or visit.hospital_discharge_location == "Death")
            and visit.encounter_time + timedelta(hours=obs_size + gap_size) <= visit.hospital_discharge_time
            and visit.hospital_discharge_time <= visit.encounter_time + timedelta(hours=obs_size + 24)
        ):
            labels[visit.visit_id] = 1
        else:
            labels[visit.visit_id] = 0
    
    return labels

@register_task("single_icu_long_term_mortality_prediction_eicu")
def single_icu_long_term_mortality_prediction_eicu_fn(
    patient: Patient,
    obs_size: int = 12,
    gap_size: int = 0,
    **kwargs
) -> Dict[str, int]:
    """TODO: to be written"""
    labels = {}

    for visit in patient:
        if (
            (visit.discharge_status == "Expired" or visit.hospital_discharge_location == "Death")
            and visit.encounter_time + timedelta(hours=obs_size + gap_size) <= visit.hospital_discharge_time
            and visit.hospital_discharge_time <= visit.encounter_time + timedelta(hours=obs_size + 336)
        ):
            labels[visit.visit_id] = 1
        else:
            labels[visit.visit_id] = 0
    
    return labels