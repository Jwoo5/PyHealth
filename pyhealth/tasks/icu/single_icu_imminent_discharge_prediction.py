from typing import Dict
from datetime import timedelta

from pyhealth.data import Patient
from pyhealth.tasks import register_task

@register_task("single_icu_imminent_discharge_prediction_mimic3")
def single_icu_imminent_discharge_prediction_mimic3_fn(
    patient: Patient,
    obs_size: int = 12,
    gap_size: int = 0,
    pred_size: int = 24,
    **kwargs
) -> Dict[str, int]:
    """TODO: to be written"""

    def categorize_imminent_dischrage_location(imminent_discharge_location: str):
        """TODO: to be written"""
        if imminent_discharge_location in [
            "DEATH",
            "DEAD/EXPIRED"
        ]:
            return 0
        elif imminent_discharge_location in [
            "HOME",
            "HOME HEALTH CARE",
            "HOME WITH HOME IV PROVIDR"
        ]:
            return 1
        elif imminent_discharge_location == "NO DISCHARGE":
            return 2
        elif imminent_discharge_location in [
            "DISC-TRAN CANCER/CHLDRN H",
            "DISC-TRAN TO FEDERAL HC",
            "DISCH-TRAN TO PSYCH HOSP",
            "HOSPICE-HOME",
            "HOSPICE-MEDICAL FACILITY",
            "ICF",
            "LEFT AGAINST MEDICAL ADVI",
            "LONG TERM CARE HOSPITAL",
            "OTHER FACILITY",
            "SHORT TERM HOSPITAL"
        ]:
            return 3
        elif imminent_discharge_location in [
            "REHAB/DISTINCT PART HOSP"
        ]:
            return 4
        elif imminent_discharge_location in [
            "SNF",
            "SNF-MEDICAID ONLY CERTIF"
        ]:
            return 5
        else:
            raise ValueError(
                "Cannot determine category number for: {}".format(imminent_discharge_location)
            )

    labels = {}

    for visit in patient:
        if (
            visit.encounter_time + timedelta(hours=obs_size + gap_size) <= visit.hospital_discharge_time
            and visit.hospital_discharge_time <= visit.encounter_time + timedelta(hours=obs_size + pred_size)
        ):
            imminent_discharge_location = visit.discharge_location
        else:
            imminent_discharge_location = "NO DISCHARGE"

        imminent_discharge_category = categorize_imminent_dischrage_location(
            imminent_discharge_location
        )

        labels[visit.visit_id] = imminent_discharge_category

    return labels

@register_task("single_icu_imminent_discharge_prediction_mimic4")
def single_icu_imminent_discharge_prediction_mimic4_fn(
    patient: Patient,
    obs_size: int = 12,
    gap_size: int = 0,
    pred_size: int = 24,
    **kwargs
) -> Dict[str, int]:
    """TODO: to be written"""
    
    def categorize_imminent_discharge_location(imminent_discharge_location: str):
        if imminent_discharge_location in [
            "Death",
            "DIED"
        ]:
            return 0
        elif imminent_discharge_location in [
            "HOME",
            "HOME HEALTH CARE"
        ]:
            return 1
        elif imminent_discharge_location == "NO DISCHARGE":
            return 2
        elif imminent_discharge_location in [
            "ACUTE HOSPITAL",
            "AGAINST ADVICE",
            "ASSISTED LIVING",
            "CHRONIC/LONG TERM ACUTE CARE",
            "HEALTHCARE FACILITY",
            "HOSPICE",
            "OTHER FACILITY",
            "PSYCH FACILITY"
        ]:
            return 3
        elif imminent_discharge_location in [
            "REHAB"
        ]:
            return 4
        elif imminent_discharge_location in [
            "SKILLED NURSING FACILITY"
        ]:
            return 5
        elif str(imminent_discharge_location) == "nan":
            return -1
        else:
            raise ValueError(
                "Cannot determine category number for: {}".format(imminent_discharge_location)
            )
    
    labels = {}
    
    for visit in patient:
        if (
            visit.encounter_time + timedelta(hours=obs_size + gap_size) <= visit.hospital_discharge_time
            and visit.hospital_discharge_time <= visit.encounter_time + timedelta(hours=obs_size + pred_size)
        ):
            imminent_discharge_location = visit.discharge_location
        else:
            imminent_discharge_location = "NO DISCHARGE"
        
        imminent_discharge_location = categorize_imminent_discharge_location(
            imminent_discharge_location
        )
        
        labels[visit.visit_id] = imminent_discharge_location
    
    return labels

@register_task("single_icu_imminent_discharge_prediction_eicu")
def single_icu_imminent_discharge_prediction_eicu(
    patient: Patient,
    obs_size: int = 12,
    gap_size: int = 0,
    pred_size: int = 24,
    **kwargs
):
    """TODO: to be written"""

    def categorize_imminent_discharge_location(imminent_discharge_location: str):
        if imminent_discharge_location == "Death":
            return 0
        elif imminent_discharge_location == "Home":
            return 1
        elif imminent_discharge_location == "NO DISCHARGE":
            return 2
        elif imminent_discharge_location in [
            "Nursing Home",
            "Other",
            "Other External",
            "Other Hospital"
        ]:
            return 3
        elif imminent_discharge_location in [
            "Rehabilitation"
        ]:
            return 4
        elif imminent_discharge_location in [
            "Skilled Nursing Facility"
        ]:
            return 5
        elif str(imminent_discharge_location) == "nan":
            return -1
        else:
            print(imminent_discharge_location)
            breakpoint()
            # raise ValueError(
            #     "Cannot determine category number for: {}".format(imminent_discharge_location)
            # )
    
    labels = {}

    for visit in patient:
        if (
            visit.encounter_time + timedelta(hours=obs_size + gap_size) <= visit.hospital_discharge_time
            and visit.hospital_discharge_time <= visit.encounter_time + timedelta(hours=obs_size + pred_size)
        ):
            imminent_discharge_location = visit.hospital_discharge_location
        else:
            imminent_discharge_location = "NO DISCHARGE"
        
        imminent_discharge_location = categorize_imminent_discharge_location(
            imminent_discharge_location
        )

        labels[visit.visit_id] = imminent_discharge_location
    
    return labels