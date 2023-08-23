from typing import Dict

from pyhealth.data import Patient
from pyhealth.tasks import register_task

@register_task("single_icu_final_acuity_prediction_mimic3")
def single_icu_final_acuity_prediction_mimic3_fn(
    patient: Patient,
    **kwargs
) -> Dict[str, int]:
    """TODO: to be written"""

    def categorize_final_acuity(final_acuity: str):
        """TODO: to be written"""
        if final_acuity in [
            "HOME",
            "HOME HEALTH CARE",
            "HOME WITH HOME IV PROVIDR",
        ]:
            return 0
        elif final_acuity == "IN-HOSPITAL MORTALITY":
            return 1
        elif final_acuity == "IN-ICU MORTALITY":
            return 2
        elif final_acuity in [
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
        elif final_acuity in [
            "REHAB/DISTINCT PART HOSP"
        ]:
            return 4
        elif final_acuity in [
            "SNF",
            "SNF-MEDICAID ONLY CERTIF"
        ]:
            return 5
        else:
            raise ValueError(
                "Cannot determine category number for: {}".format(final_acuity)
            )

    labels = {}

    for visit in patient:
        if (
            visit.hospital_discharge_time <= visit.discharge_time
            and visit.discharge_location in [
                "Death",
                "DEAD/EXPIRED"
            ]
        ):
            final_acuity_label = "IN-ICU MORTALITY"
        elif visit.discharge_location in [
            "Death",
            "DEAD/EXPIRED"
        ]:
            final_acuity_label = "IN-HOSPITAL MORTALITY"
        else:
            final_acuity_label = visit.discharge_location
        
        final_acuity_category = categorize_final_acuity(final_acuity_label)
        
        labels[visit.visit_id] = final_acuity_category

    return labels

@register_task("single_icu_final_acuity_prediction_mimic4")
def single_icu_final_acuity_prediction_mimic4_fn(
    patient: Patient,
    **kwargs
) -> Dict[str, int]:
    """TODO: to be written"""
    
    def categorize_final_acuity(final_acuity: str):
        if final_acuity in [
            "HOME",
            "HOME HEALTH CARE"
        ]:
            return 0
        elif final_acuity == "IN-HOSPITAL MORTALITY":
            return 1
        elif final_acuity == "IN-ICU MORTALITY":
            return 2
        elif final_acuity in [
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
        elif final_acuity in [
            "REHAB"
        ]:
            return 4
        elif final_acuity in [
            "SKILLED NURSING FACILITY"
        ]:
            return 5
        elif str(final_acuity) == "nan":
            return -1
        else:
            raise ValueError(
                "Cannot determine category number for: {}".format(final_acuity)
            )
    
    labels = {}
    
    for visit in patient:
        if (
            visit.hospital_discharge_time <= visit.discharge_time
            and visit.discharge_location in [
                "Death",
                "DIED"
            ]
        ):
            final_acuity_label = "IN-ICU MORTALITY"
        elif visit.discharge_location in [
            "Death",
            "DIED"
        ]:
            final_acuity_label = "IN-HOSPITAL MORTALITY"
        else:
            final_acuity_label = visit.discharge_location
        
        final_acuity_category = categorize_final_acuity(final_acuity_label)
        
        labels[visit.visit_id] = final_acuity_category
    
    return labels

@register_task("single_icu_final_acuity_prediction_eicu")
def single_icu_final_acuity_prediction_eicu_fn(
    patient: Patient,
    **kwargs
) -> Dict[str, int]:
    """TODO: to be written"""

    def categorize_final_acuity(final_acuity: str):
        if final_acuity in [
            "Home"
        ]:
            return 0
        elif final_acuity == "IN-HOSPITAL MORTALITY":
            return 1
        elif final_acuity == "IN-ICU MORTALITY":
            return 2
        elif final_acuity in [
            "Nursing Home",
            "Other",
            "Other External",
            "Other Hospital",
        ]:
            return 3
        elif final_acuity in [
            "Rehabilitation"
        ]:
            return 4
        elif final_acuity in [
            "Skilled Nursing Facility"
        ]:
            return 5
        elif str(final_acuity) == "nan":
            return -1
        else:
            raise ValueError(
                "Cannot determine category number for: {}".format(final_acuity)
            )
    
    labels = {}

    for visit in patient:
        if visit.discharge_status == "Expired":
            final_acuity_label = "IN-ICU MORTALITY"
        elif visit.hospital_discharge_location == "Death":
            final_acuity_label = "IN-HOSPITAL MORTALITY"
        else:
            final_acuity_label = visit.hospital_discharge_location
        
        final_acuity_category = categorize_final_acuity(final_acuity_label)

        labels[visit.visit_id] = final_acuity_category
    
    return labels