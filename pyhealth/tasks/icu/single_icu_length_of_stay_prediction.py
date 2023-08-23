from typing import Dict

from pyhealth.data import Patient
from pyhealth.tasks import register_task

@register_task("single_icu_length_of_stay_3_prediction_mimic3")
def single_icu_length_of_stay_3_prediction_mimic3_fn(
    patient: Patient,
    **kwargs
) -> Dict[str, int]:
    """TODO: to be written"""
    labels = {}
    
    for visit in patient:
        los_days = (visit.discharge_time - visit.encounter_time).days
        labels[visit.visit_id] = 1 if los_days >= 3 else 0
    
    return labels

@register_task("single_icu_length_of_stay_7_prediction_mimic3")
def single_icu_length_of_stay_7_prediction_mimic3_fn(
    patient: Patient,
    **kwargs
) -> Dict[str, int]:
    """TODO: to be written"""
    labels = {}
    
    for visit in patient:
        los_days = (visit.discharge_time - visit.encounter_time).days
        labels[visit.visit_id] = 1 if los_days >= 7 else 0

    return labels

@register_task("single_icu_length_of_stay_3_prediction_mimic4")
def single_icu_length_of_stay_3_prediction_mimic4_fn(
    patient: Patient,
    **kwargs
) -> Dict[str, int]:
    """TODO: to be written"""
    labels = {}
    
    for visit in patient:
        los_days = (visit.discharge_time - visit.encounter_time).days
        labels[visit.visit_id] = 1 if los_days >= 3 else 0
    
    return labels

@register_task("single_icu_length_of_stay_7_prediction_mimic4")
def single_icu_length_of_stay_7_prediction_mimic4_fn(
    patient: Patient,
    **kwargs
) -> Dict[str, int]:
    """TODO: to be written"""
    labels = {}
    
    for visit in patient:
        los_days = (visit.discharge_time - visit.encounter_time).days
        labels[visit.visit_id] = 1 if los_days >= 7 else 0
    
    return labels

@register_task("single_icu_length_of_stay_3_prediction_eicu")
def single_icu_length_of_stay_3_prediction_eicu_fn(
    patient: Patient,
    **kwargs,
) -> Dict[str, int]:
    """TODO: to be written"""
    labels = {}

    for visit in patient:
        los_days = (visit.discharge_time - visit.encounter_time).days
        labels[visit.visit_id] = 1 if los_days >= 3 else 0
    
    return labels

@register_task("single_icu_length_of_stay_7_prediction_eicu")
def single_icu_length_of_stay_7_prediction_eicu_fn(
    patient: Patient,
    **kwargs
) -> Dict[str, int]:
    """TODO: to be written"""
    labels = {}

    for visit in patient:
        los_days = (visit.discharge_time - visit.encounter_time).days
        labels[visit.visit_id] = 1 if los_days >= 7 else 0
    
    return labels