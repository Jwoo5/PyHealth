from collections import OrderedDict
from typing import Dict

from pyhealth.data import Patient, Visit
from pyhealth.tasks import register_task

@register_task("single_icu_readmission_prediction_mimic3")
def single_icu_readmission_prediction_mimic3_fn(
    patient: Patient,
    **kwargs
) -> Dict[str, int]:
    """TODO: to be written"""
    labels = {}

    # we need to sort visits by encounter time
    visits = list(
        sorted(
            patient.visits.values(), key=lambda x: (x.hadm_id, x.encounter_time), reverse=False
        )
    )

    for i, visit in enumerate(visits):
        visit: Visit

        if i + 1 == len(visits):
            labels[visit.visit_id] = 0
            break

        next_visit: Visit = visits[i + 1]

        if visit.hadm_id == next_visit.hadm_id:
            assert visit.discharge_time <= next_visit.encounter_time
            labels[visit.visit_id] = 1
        else:
            labels[visit.visit_id] = 0

    return labels

@register_task("single_icu_readmission_prediction_mimic4")
def single_icu_readmission_prediction_mimic4_fn(
    patient: Patient,
    **kwargs
) -> Dict[str, int]:
    """TODO: to be written"""
    labels = {}
    
    # we need to sort visits by encounter time
    visits = list(
        sorted(
            patient.visits.values(), key=lambda x: (x.hadm_id, x.encounter_time), reverse=False
        )
    )
    
    for i, visit in enumerate(visits):
        visit: Visit
        
        if i + 1 == len(visits):
            labels[visit.visit_id] = 0
            break
        
        next_visit: Visit = visits[i + 1]
        
        if visit.hadm_id == next_visit.hadm_id:
            assert visit.discharge_time <= next_visit.encounter_time
            labels[visit.visit_id] = 1
        else:
            labels[visit.visit_id] = 0
    
    return labels

@register_task("single_icu_readmission_prediction_eicu")
def single_icu_readmission_prediction_eicu_fn(
    patient: Patient,
    **kwargs
) -> Dict[str, int]:
    """TODO: to be written"""
    labels = {}

    # we need to sort visits by visit number
    visits = list(
        sorted(
            patient.visits.values(), key=lambda x: (x.hadm_id, x.visit_number), reverse=False
        )
    )

    for i, visit in enumerate(visits):
        visit: Visit

        if i + 1 == len(visits):
            labels[visit.visit_id] = 0
            break

        next_visit: Visit = visits[i + 1]

        if visit.hadm_id == next_visit.hadm_id:
            assert visit.discharge_time <= next_visit.encounter_time
            labels[visit.visit_id] = 1
        else:
            labels[visit.visit_id] = 0
    
    return labels