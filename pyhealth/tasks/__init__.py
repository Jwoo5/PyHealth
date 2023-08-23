import os
import importlib
from typing import Callable

from .drug_recommendation import (
    drug_recommendation_eicu_fn,
    drug_recommendation_mimic3_fn,
    drug_recommendation_mimic4_fn,
    drug_recommendation_omop_fn,
)
from .cardiology_detect import (
    cardiology_isAR_fn,
    cardiology_isBBBFB_fn,
    cardiology_isAD_fn,
    cardiology_isCD_fn,
    cardiology_isWA_fn,
)
from .length_of_stay_prediction import (
    length_of_stay_prediction_eicu_fn,
    length_of_stay_prediction_mimic3_fn,
    length_of_stay_prediction_mimic4_fn,
    length_of_stay_prediction_omop_fn,
)
from .mortality_prediction import (
    mortality_prediction_eicu_fn,
    mortality_prediction_eicu_fn2,
    mortality_prediction_mimic3_fn,
    mortality_prediction_mimic4_fn,
    mortality_prediction_omop_fn,
)
from .readmission_prediction import (
    readmission_prediction_eicu_fn,
    readmission_prediction_eicu_fn2,
    readmission_prediction_mimic3_fn,
    readmission_prediction_mimic4_fn,
    readmission_prediction_omop_fn,
)
from .sleep_staging import (
    sleep_staging_sleepedf_fn,
    sleep_staging_isruc_fn,
    sleep_staging_shhs_fn,
)


# registry
TASK_REGISTRY = {}

def register_task(name):
    """
    New tasks can be added to pyhealth with the
    :func:`~pyhealth.tasks.register_task` function decorator
    
    For example::

        @register_task("icu_mortality_prediction_mimic3")
        def icu_mortality_prediction_mimic3_fn(...):
            (...)
    
    Args:
        name (str): the name of the task
    """
    
    def register_task_fn(fn: Callable):
        if name in TASK_REGISTRY:
            raise ValueError("Cannot register duplicate task ({})".format(name))
        TASK_REGISTRY[name] = fn
        
        return fn
    return register_task_fn

def get_task(name):
    return TASK_REGISTRY[name]

def import_tasks(tasks_dir, namespace):
    for file in os.listdir(tasks_dir):
        path = os.path.join(tasks_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            task_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + task_name)

# automatically import any Python files in the tasks/ directory
tasks_dir = os.path.dirname(__file__)
import_tasks(tasks_dir, "pyhealth.tasks")