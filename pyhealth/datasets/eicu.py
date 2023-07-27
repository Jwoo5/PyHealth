import os
from typing import Optional, List, Dict, Tuple, Union

import pandas as pd
from tqdm import tqdm
from datetime import datetime

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, ArrayType, StringType

from pyhealth.data import Event, Visit, Patient
from pyhealth.datasets import BaseEHRDataset, BaseEHRSparkDataset
from pyhealth.datasets.utils import strptime, padyear

# TODO: add other tables


class eICUDataset(BaseEHRDataset):
    """Base dataset for eICU dataset.

    The eICU dataset is a large dataset of de-identified health records of ICU
    patients. The dataset is available at https://eicu-crd.mit.edu/.

    The basic information is stored in the following tables:
        - patient: defines a patient (uniquepid), a hospital admission
            (patienthealthsystemstayid), and a ICU stay (patientunitstayid)
            in the database.
        - hospital: contains information about a hospital (e.g., region).

    Note that in eICU, a patient can have multiple hospital admissions and each
    hospital admission can have multiple ICU stays. The data in eICU is centered
    around the ICU stay and all timestamps are relative to the ICU admission time.
    Thus, we only know the order of ICU stays within a hospital admission, but not
    the order of hospital admissions within a patient. As a result, we use `Patient`
    object to represent a hospital admission of a patient, and use `Visit` object to
    store the ICU stays within that hospital admission.

    We further support the following tables:
        - diagnosis: contains ICD diagnoses (ICD9CM and ICD10CM code)
            and diagnosis information (under attr_dict) for patients
        - treatment: contains treatment information (eICU_TREATMENTSTRING code)
            for patients.
        - medication: contains medication related order entries (eICU_DRUGNAME
            code) for patients.
        - lab: contains laboratory measurements (eICU_LABNAME code)
            for patients
        - physicalExam: contains all physical exam (eICU_PHYSICALEXAMPATH)
            conducted for patients.
        - admissionDx:  table contains the primary diagnosis for admission to
            the ICU per the APACHE scoring criteria. (eICU_ADMITDXPATH)

    Args:
        dataset_name: name of the dataset.
        root: root directory of the raw data (should contain many csv files).
        tables: list of tables to be loaded (e.g., ["DIAGNOSES_ICD", "PROCEDURES_ICD"]).
        code_mapping: a dictionary containing the code mapping information.
            The key is a str of the source code vocabulary and the value is of
            two formats:
                (1) a str of the target code vocabulary;
                (2) a tuple with two elements. The first element is a str of the
                    target code vocabulary and the second element is a dict with
                    keys "source_kwargs" or "target_kwargs" and values of the
                    corresponding kwargs for the `CrossMap.map()` method.
            Default is empty dict, which means the original code will be used.
        dev: whether to enable dev mode (only use a small subset of the data).
            Default is False.
        refresh_cache: whether to refresh the cache; if true, the dataset will
            be processed from scratch and the cache will be updated. Default is False.

    Attributes:
        task: Optional[str], name of the task (e.g., "mortality prediction").
            Default is None.
        samples: Optional[List[Dict]], a list of samples, each sample is a dict with
            patient_id, visit_id, and other task-specific attributes as key.
            Default is None.
        patient_to_index: Optional[Dict[str, List[int]]], a dict mapping patient_id to
            a list of sample indices. Default is None.
        visit_to_index: Optional[Dict[str, List[int]]], a dict mapping visit_id to a
            list of sample indices. Default is None.

    Examples:
        >>> from pyhealth.datasets import eICUDataset
        >>> dataset = eICUDataset(
        ...         root="/srv/local/data/physionet.org/files/eicu-crd/2.0",
        ...         tables=["diagnosis", "medication", "lab", "treatment", "physicalExam", "admissionDx"],
        ...     )
        >>> dataset.stat()
        >>> dataset.info()
    """

    def __init__(self, **kwargs):
        # store a mapping from visit_id to patient_id
        # will be used to parse clinical tables as they only contain visit_id
        self.visit_id_to_patient_id: Dict[str, str] = {}
        self.visit_id_to_encounter_time: Dict[str, datetime] = {}
        super(eICUDataset, self).__init__(**kwargs)

    def parse_basic_info(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper functions which parses patient and hospital tables.

        Will be called in `self.parse_tables()`.

        Docs:
            - patient: https://eicu-crd.mit.edu/eicutables/patient/
            - hospital: https://eicu-crd.mit.edu/eicutables/hospital/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.

        Note:
            We use `Patient` object to represent a hospital admission of a patient,
            and use `Visit` object to store the ICU stays within that hospital
            admission.
        """
        # read patient table
        patient_df = pd.read_csv(
            os.path.join(self.root, "patient.csv"),
            dtype={
                "uniquepid": str,
                "patienthealthsystemstayid": str,
                "patientunitstayid": str,
            },
            nrows=5000 if self.dev else None,
        )
        # read hospital table
        hospital_df = pd.read_csv(os.path.join(self.root, "hospital.csv"))
        hospital_df.region = hospital_df.region.fillna("Unknown").astype(str)
        # merge patient and hospital tables
        df = pd.merge(patient_df, hospital_df, on="hospitalid", how="left")
        # sort by ICU admission and discharge time
        df["neg_hospitaladmitoffset"] = -df["hospitaladmitoffset"]
        df = df.sort_values(
            [
                "uniquepid",
                "patienthealthsystemstayid",
                "neg_hospitaladmitoffset",
                "unitdischargeoffset",
            ],
            ascending=True,
        )
        # group by patient and hospital admission
        df_group = df.groupby(["uniquepid", "patienthealthsystemstayid"])
        # load patients
        for (p_id, ha_id), p_info in tqdm(df_group, desc="Parsing patients"):
            # each Patient object is a single hospital admission of a patient
            patient_id = f"{p_id}+{ha_id}"

            # hospital admission time (Jan 1 of hospitaldischargeyear, 00:00:00)
            ha_datetime = strptime(padyear(str(p_info["hospitaldischargeyear"].values[0])))

            # no exact birth datetime in eICU
            # use hospital admission time and age to approximate birth datetime
            age = p_info["age"].values[0]
            if pd.isna(age):
                birth_datetime = None
            elif age == "> 89":
                birth_datetime = ha_datetime - pd.DateOffset(years=89)
            else:
                birth_datetime = ha_datetime - pd.DateOffset(years=int(age))

            # no exact death datetime in eICU
            # use hospital discharge time to approximate death datetime
            death_datetime = None
            if p_info["hospitaldischargestatus"].values[0] == "Expired":
                ha_los_min = (
                    p_info["hospitaldischargeoffset"].values[0]
                    - p_info["hospitaladmitoffset"].values[0]
                )
                death_datetime = ha_datetime + pd.Timedelta(minutes=ha_los_min)

            patient = Patient(
                patient_id=patient_id,
                birth_datetime=birth_datetime,
                death_datetime=death_datetime,
                gender=p_info["gender"].values[0],
                ethnicity=p_info["ethnicity"].values[0],
            )

            # load visits
            for v_id, v_info in p_info.groupby("patientunitstayid"):
                # each Visit object is a single ICU stay within a hospital admission

                # base time is the hospital admission time
                unit_admit = v_info["neg_hospitaladmitoffset"].values[0]
                unit_discharge = unit_admit + v_info["unitdischargeoffset"].values[0]
                encounter_time = ha_datetime + pd.Timedelta(minutes=unit_admit)
                discharge_time = ha_datetime + pd.Timedelta(minutes=unit_discharge)

                visit = Visit(
                    visit_id=v_id,
                    patient_id=patient_id,
                    encounter_time=encounter_time,
                    discharge_time=discharge_time,
                    discharge_status=v_info["unitdischargestatus"].values[0],
                    hospital_id=v_info["hospitalid"].values[0],
                    region=v_info["region"].values[0],
                )

                # add visit
                patient.add_visit(visit)
                # add visit id to patient id mapping
                self.visit_id_to_patient_id[v_id] = patient_id
                # add visit id to encounter time mapping
                self.visit_id_to_encounter_time[v_id] = encounter_time
            # add patient
            patients[patient_id] = patient
        return patients

    def parse_diagnosis(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses diagnosis table.

        Will be called in `self.parse_tables()`.

        Docs:
            - diagnosis: https://eicu-crd.mit.edu/eicutables/diagnosis/

        Args:
            patients: a dict of Patient objects indexed by patient_id.

        Returns:
            The updated patients dict.

        Note:
            This table contains both ICD9CM and ICD10CM codes in one single
                cell. We need to use medcode to distinguish them.
        """

        # load ICD9CM and ICD10CM coding systems
        from pyhealth.medcode import ICD9CM, ICD10CM

        icd9cm = ICD9CM()
        icd10cm = ICD10CM()

        def icd9cm_or_icd10cm(code):
            if code in icd9cm:
                return "ICD9CM"
            elif code in icd10cm:
                return "ICD10CM"
            else:
                return "Unknown"

        table = "diagnosis"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"patientunitstayid": str, "icd9code": str, "diagnosisstring": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["patientunitstayid", "icd9code", "diagnosisstring"])
        # sort by diagnosisoffset
        df = df.sort_values(["patientunitstayid", "diagnosisoffset"], ascending=True)
        # add the patient id info
        df["patient_id"] = df["patientunitstayid"].apply(
            lambda x: self.visit_id_to_patient_id.get(x, None)
        )
        # add the visit encounter time info
        df["v_encounter_time"] = df["patientunitstayid"].apply(
            lambda x: self.visit_id_to_encounter_time.get(x, None)
        )
        # group by visit
        group_df = df.groupby("patientunitstayid")

        # parallel unit of diagnosis (per visit)
        def diagnosis_unit(v_info):
            v_id = v_info["patientunitstayid"].values[0]
            patient_id = v_info["patient_id"].values[0]
            v_encounter_time = v_info["v_encounter_time"].values[0]
            if patient_id is None:
                return []

            events = []
            for offset, codes, dxstr in zip(v_info["diagnosisoffset"], v_info["icd9code"],
                                            v_info["diagnosisstring"]):
                # compute the absolute timestamp
                timestamp = v_encounter_time + pd.Timedelta(minutes=offset)
                codes = [c.strip() for c in codes.split(",")]
                # for each code in a single cell (mixed ICD9CM and ICD10CM)
                for code in codes:
                    vocab = icd9cm_or_icd10cm(code)
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary=vocab,
                        visit_id=v_id,
                        patient_id=patient_id,
                        timestamp=timestamp,
                        diagnosisString=dxstr
                    )
                    # update patients
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(lambda x: diagnosis_unit(x))

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def parse_treatment(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses treatment table.

        Will be called in `self.parse_tables()`.

        Docs:
            - treatment: https://eicu-crd.mit.edu/eicutables/treatment/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "treatment"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"patientunitstayid": str, "treatmentstring": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["patientunitstayid", "treatmentstring"])
        # sort by treatmentoffset
        df = df.sort_values(["patientunitstayid", "treatmentoffset"], ascending=True)
        # add the patient id info
        df["patient_id"] = df["patientunitstayid"].apply(
            lambda x: self.visit_id_to_patient_id.get(x, None)
        )
        # add the visit encounter time info
        df["v_encounter_time"] = df["patientunitstayid"].apply(
            lambda x: self.visit_id_to_encounter_time.get(x, None)
        )
        # group by visit
        group_df = df.groupby("patientunitstayid")

        # parallel unit of treatment (per visit)
        def treatment_unit(v_info):
            v_id = v_info["patientunitstayid"].values[0]
            patient_id = v_info["patient_id"].values[0]
            v_encounter_time = v_info["v_encounter_time"].values[0]
            if patient_id is None:
                return []

            events = []
            for offset, code in zip(
                v_info["treatmentoffset"], v_info["treatmentstring"]
            ):
                # compute the absolute timestamp
                timestamp = v_encounter_time + pd.Timedelta(minutes=offset)
                event = Event(
                    code=code,
                    table=table,
                    vocabulary="eICU_TREATMENTSTRING",
                    visit_id=v_id,
                    patient_id=patient_id,
                    timestamp=timestamp,
                )
                # update patients
                events.append(event)

            return events

        # parallel apply
        group_df = group_df.parallel_apply(lambda x: treatment_unit(x))

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def parse_medication(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses medication table.

        Will be called in `self.parse_tables()`.

        Docs:
            - medication: https://eicu-crd.mit.edu/eicutables/medication/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "medication"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            low_memory=False,
            dtype={"patientunitstayid": str, "drugname": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["patientunitstayid", "drugname"])
        # sort by drugstartoffset
        df = df.sort_values(["patientunitstayid", "drugstartoffset"], ascending=True)
        # add the patient id info
        df["patient_id"] = df["patientunitstayid"].apply(
            lambda x: self.visit_id_to_patient_id.get(x, None)
        )
        # add the visit encounter time info
        df["v_encounter_time"] = df["patientunitstayid"].apply(
            lambda x: self.visit_id_to_encounter_time.get(x, None)
        )
        # group by visit
        group_df = df.groupby("patientunitstayid")

        # parallel unit of medication (per visit)
        def medication_unit(v_info):
            v_id = v_info["patientunitstayid"].values[0]
            patient_id = v_info["patient_id"].values[0]
            v_encounter_time = v_info["v_encounter_time"].values[0]
            if patient_id is None:
                return []

            events = []
            for offset, code in zip(v_info["drugstartoffset"], v_info["drugname"]):
                # compute the absolute timestamp
                timestamp = v_encounter_time + pd.Timedelta(minutes=offset)
                event = Event(
                    code=code,
                    table=table,
                    vocabulary="eICU_DRUGNAME",
                    visit_id=v_id,
                    patient_id=patient_id,
                    timestamp=timestamp,
                )
                # update patients
                events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(lambda x: medication_unit(x))

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def parse_lab(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses lab table.

        Will be called in `self.parse_tables()`.

        Docs:
            - lab: https://eicu-crd.mit.edu/eicutables/lab/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "lab"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"patientunitstayid": str, "labname": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["patientunitstayid", "labname"])
        # sort by labresultoffset
        df = df.sort_values(["patientunitstayid", "labresultoffset"], ascending=True)
        # add the patient id info
        df["patient_id"] = df["patientunitstayid"].apply(
            lambda x: self.visit_id_to_patient_id.get(x, None)
        )
        # add the visit encounter time info
        df["v_encounter_time"] = df["patientunitstayid"].apply(
            lambda x: self.visit_id_to_encounter_time.get(x, None)
        )
        # group by visit
        group_df = df.groupby("patientunitstayid")

        # parallel unit of lab (per visit)
        def lab_unit(v_info):
            v_id = v_info["patientunitstayid"].values[0]
            patient_id = v_info["patient_id"].values[0]
            v_encounter_time = v_info["v_encounter_time"].values[0]
            if patient_id is None:
                return []

            events = []
            for offset, code in zip(v_info["labresultoffset"], v_info["labname"]):
                # compute the absolute timestamp
                timestamp = v_encounter_time + pd.Timedelta(minutes=offset)
                event = Event(
                    code=code,
                    table=table,
                    vocabulary="eICU_LABNAME",
                    visit_id=v_id,
                    patient_id=patient_id,
                    timestamp=timestamp,
                )
                # update patients
                events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(lambda x: lab_unit(x))

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def parse_physicalexam(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses physicalExam table.

        Will be called in `self.parse_tables()`.

        Docs:
            - physicalExam: https://eicu-crd.mit.edu/eicutables/physicalexam/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "physicalExam"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"patientunitstayid": str, "physicalexampath": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["patientunitstayid", "physicalexampath"])
        # sort by treatmentoffset
        df = df.sort_values(["patientunitstayid", "physicalexamoffset"], ascending=True)
        # add the patient id info
        df["patient_id"] = df["patientunitstayid"].apply(
            lambda x: self.visit_id_to_patient_id.get(x, None)
        )
        # add the visit encounter time info
        df["v_encounter_time"] = df["patientunitstayid"].apply(
            lambda x: self.visit_id_to_encounter_time.get(x, None)
        )
        # group by visit
        group_df = df.groupby("patientunitstayid")

        # parallel unit of physicalExam (per visit)
        def physicalExam_unit(v_info):
            v_id = v_info["patientunitstayid"].values[0]
            patient_id = v_info["patient_id"].values[0]
            v_encounter_time = v_info["v_encounter_time"].values[0]
            if patient_id is None:
                return []

            events = []
            for offset, code in zip(
                v_info["physicalexamoffset"], v_info["physicalexampath"]
            ):
                # compute the absolute timestamp
                timestamp = v_encounter_time + pd.Timedelta(minutes=offset)
                event = Event(
                    code=code,
                    table=table,
                    vocabulary="eICU_PHYSICALEXAMPATH",
                    visit_id=v_id,
                    patient_id=patient_id,
                    timestamp=timestamp,
                )
                # update patients
                events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(lambda x: physicalExam_unit(x))

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

class eICUSparkDataset(BaseEHRSparkDataset):
    """Base dataset for eICU dataset utilized by PySpark for the efficient data processing.
    
    The eICU dataset is a large dataset of de-identified health records of ICU
    patients. The dataset is available at https://eicu-crd.mit.edu/.

    Unlike the normal eicu dataset (eICUDataset) that does not utilize PySpark, this
    dataset provides not only the corresponding code for each event but also any other
    available columns by attr_dict or internal attributes to support further data processing.

    The basic information is stored in the following tables:
        - patient: defines a patient (uniquepid), a hospital admission
            (patienthealthsystemstayid), and a ICU stay (patientunitstayid)
            in the database.
        - hospital: contains information about a hospital (e.g., region).

    Note that in eICU, a patient can have multiple hospital admissions and each
    hospital admission can have multiple ICU stays. The data in eICU is centered
    around the ICU stay and all timestamps are relative to the ICU admission time.
    Thus, we only know the order of ICU stays within a hospital admission, but not
    the order of hospital admissions within a patient. As a result, we use `Patient`
    object to represent a hospital admission of a patient, and use `Visit` object to
    store the ICU statys within that hospital admission.
    
    We further support the following tables:
        - diagnosis: contains ICD diagnoses (ICD9CM and ICD10CM code)
            and diagnosis information (under attr_dict) for patients.
        - treatment: contains treatment information (eICU_TREATMENTSTRING code)
            for patients.
        - medication: contains medication related order entries (eICU_DRUGNAME
            code) for patients.
        - lab: contains laboratory measurements (eICU_LABNAME code) for patients
        - physicalExam: contains all physical exam (eICU_PHYSICALEXAMPATH) conducted
            for patients.
        - admissionDx: table contains the primary diagnosis for admission to
            the ICU per the APACHE scoring criteria. (eICU_ADMITDXPATH)
        - infusionDrug: contains details of drug infusions (eICU_DRUGNAME code)
            for patients.
        - nurseCare: contains the categories for nurses to document patient care
            information (eicu_CELLATTRIBUTEPATH code) for patients.
        - nurseCharting: contains information entered by the nurse
            (eICU_NURSINGCHARTCELLTYPEVALNAME code) for patients.
        - intakeOutput: contains intake and output (eicu_INTAKEOUTPUTCELLPATH code)
            recorded for patients.
        - microLab: contains microbiology data (eICU_MICROLABSTRING) for patients.
        - nurseAssessment: contains the information to assess and document patient
            items by nurses (eICU_CELLATTRIBUTEPATH code) for patients.
        - vitalPeriodic: contains data which is consistently interfaced from bedside
            vital signs monitors (eICU_VITALPERIODICSTRING code) for patients. Note
            that we set `code` to `None` for all events from this table because there
            is no specific column for `code` in vitalPeriodic table.
        - vitalAperiodic: contains invasive vital sign data at irregular intervals
            (eICU_VITALPERIODICSTRING code) for patients. Similar to vitalPeriodic table,
            we set `code` to `None` for all events from this table.

    Args:
        dataset_name: name of the dataset.
        root: root directory of the raw data (should contain many csv files).
        tables: list of tables to be loaded (e.g., ["lab", "infusiondrug"]). Basic
            tables will be loaded by default.
        visit_unit: unit of visit to be grouped. Available options are typed in VISIT_UNIT_CHOICES.
            Default is "icu", which means to regard each ICU admission as a visit. only "icu" is
            allowed for eICU dataset.
        observation_size: size of the observation window. only the events within the first N hours
            are included in the processed data. Default is 12.
        gap_size: size of gap window. labels of some prediction tasks (E.g., short-term mortality
            prediction) are defined between `observation_size` + `gap_size` and the next N hours of
            `prediction_size`. If a patient's stay has less than `observation_size` + `gap_size`
            duration, some tasks cannot be defined for that stay. Default is 0.
        prediction_size: size of prediction window. labels of some prediction tasks (E.g., short-term
            mortality prediction) are defined between `observation_size` + `gap_size` and the next N
            hours of `prediction_size`. Default is 24.
        code_mapping: a dictionary containing the code mapping information.
            The key is a str of the source code vocabulary and the value is of
            two formats:
                - a str of the target code vocabulry. E.g., {"NDC", "ATC"}.
                - a tuple with two elements. The first element is a str of the
                    target code vocabulary and the second element is a dict with
                    keys "source_kwargs" or "target_kwargs" and values of the
                    corresponding kwargs for the `CrossMap.map()` method. E.g.,
                    {"NDC", ("ATC", {"target_kwargs": {"level": 3}})}.
                Default is empty dict, which means the original code will be used.
        dev: whether to enable dev mode (only use a small subset of the data).
            Default is False.
        refresh_cache: whether to refresh the cache; if true, the dataset will
            be processed from scratch and cache will be updated. Default is False.
    
    Examples:
        >>> from pyhealth.datasets import eICUSparkDataset
        >>> dataset = eICUSparkDataset(
        ...     root="/usr/local/data/physionet.org/files/eicu-crd/2.0",
        ...     tables=["diagnosis", "medication", "lab", "treatment", "physicalExam", "admissionDx"],
        ...     visit_unit="icu",
        ...     observation_size=12,
        ...     gap_size=0,
        ...     prediction_size=24,
        ... )
        >>> dataset.stat()
        >>> dataset.info()
    """

    def __init__(
        self,
        root,
        tables,
        visit_unit="icu",
        **kwargs
    ):
        # store a mapping from visit_id to patient_id
        # will be used to parse clinical tables as they only contain visit_id
        self.visit_id_to_patient_id: Dict[str, str] = {}
        self.visit_id_to_encounter_time: Dict[str, datetime] = {}

        assert visit_unit == "icu", (
            "We only allow icu visit level in eicu since the order of hospital visits "
            "is not determinable in eicu."
        )
        self.visit_key = "patientunitstayid"
        # eicu does not provide encounter information
        self.encounter_key = None
        self.discharge_key = "unitdischargeoffset"

        super().__init__(root, tables, visit_unit, **kwargs)

    def parse_basic_info(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper functions which parses patient and hospital tables.

        Will be called in `self.parse_tables()`.

        Docs:
            - patient: https://eicu-crd.mit.edu/eicutables/patient/
            - hospital: https://eicu-crd.mit.edu/eicutables/hospital/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.

        Note:
            We use `Patient` object to represent a hospital admission of a patient,
            and use `Visit` object to store the ICU stays within that hospital
            admission.
        """
        # read patient table
        patient_df = pd.read_csv(
            os.path.join(self.root, "patient.csv"),
            dtype={
                "uniquepid": str,
                "patienthealthsystemstayid": str,
                "patientunitstayid": str,
            },
            nrows=5000 if self.dev else None,
        )
        # read hospital table
        hospital_df = pd.read_csv(os.path.join(self.root, "hospital.csv"))
        hospital_df.region = hospital_df.region.fillna("Unknown").astype(str)
        # merge patient and hospital tables
        df = pd.merge(patient_df, hospital_df, on="hospitalid", how="left")
        # sort by ICU admission and discharge time
        df["neg_hospitaladmitoffset"] = -df["hospitaladmitoffset"]
        df = df.sort_values(
            [
                "uniquepid",
                "patienthealthsystemstayid",
                "neg_hospitaladmitoffset",
                "unitdischargeoffset",
            ],
            ascending=True,
        )
        # group by patient and hospital admission
        df_group = df.groupby(["uniquepid", "patienthealthsystemstayid"])
        # load patients
        for (p_id, ha_id), p_info in tqdm(df_group, desc="Parsing patients"):
            # each Patient object is a single hospital admission of a patient
            patient_id = f"{p_id}+{ha_id}"

            # hospital admission time (Jan 1 of hospitaldischargeyear, 00:00:00)
            ha_datetime = strptime(str(p_info["hospitaldischargeyear"].values[0]))

            # no exact birth datetime in eICU
            # use hospital admission time and age to approximate birth datetime
            age = p_info["age"].values[0]
            if pd.isna(age):
                birth_datetime = None
            elif age == "> 89":
                birth_datetime = ha_datetime - pd.DateOffset(years=89)
            else:
                birth_datetime = ha_datetime - pd.DateOffset(years=int(age))

            # no exact death datetime in eICU
            # use hospital discharge time to approximate death datetime
            death_datetime = None
            if p_info["hospitaldischargestatus"].values[0] == "Expired":
                ha_los_min = (
                    p_info["hospitaldischargeoffset"].values[0]
                    - p_info["hospitaladmitoffset"].values[0]
                )
                death_datetime = ha_datetime + pd.Timedelta(minutes=ha_los_min)

            patient = Patient(
                patient_id=patient_id,
                birth_datetime=birth_datetime,
                death_datetime=death_datetime,
                gender=p_info["gender"].values[0],
                ethnicity=p_info["ethnicity"].values[0],
            )

            # load visits
            for v_id, v_info in p_info.groupby("patientunitstayid"):
                # each Visit object is a single ICU stay within a hospital admission

                # base time is the hospital admission time
                unit_admit = v_info["neg_hospitaladmitoffset"].values[0]
                unit_discharge = unit_admit + v_info["unitdischargeoffset"].values[0]
                encounter_time = ha_datetime + pd.Timedelta(minutes=unit_admit)
                discharge_time = ha_datetime + pd.Timedelta(minutes=unit_discharge)
                
                visit = Visit(
                    visit_id=v_id,
                    patient_id=patient_id,
                    encounter_time=encounter_time,
                    discharge_time=discharge_time,
                    discharge_status=v_info["unitdischargestatus"].values[0],
                    hospital_id=v_info["hospitalid"].values[0],
                    region=v_info["region"].values[0],
                )

                # add visit
                patient.add_visit(visit)
                # add visit id to patient id mapping
                self.visit_id_to_patient_id[v_id] = patient_id
                # add visit id to encounter time mapping
                self.visit_id_to_encounter_time[v_id] = encounter_time
            # add patient
            patients[patient_id] = patient
        return patients

    def parse_diagnosis(
        self,
        patients: Dict[str, Patient],
        spark: SparkSession
    ) -> Dict[str, Patient]:
        """Helper function which parses diagnosis table.
        
        Will be called in `self.parse_tables()`.
        
        Docs:
            - diagnosis: https://eicu-crd.mit.edu/eicutables/diagnosis/
        
        Args:
            patients: a dict of Patient objects indexed by patient_id.
            spark: a spark session for reading the table.
        
        Returns:
            The updated patients dict.
        
        Note:
            This table contains both ICD9CM and ICD10CM codes in one single
                cell. We need to use medcode to distinguish them.
        """
        
        # load ICD9CM and ICD10CM coding systems
        from pyhealth.medcode import ICD9CM, ICD10CM
        
        icd9cm = ICD9CM()
        icd10cm = ICD10CM()
        
        def icd9cm_or_icd10cm(code):
            if code in icd9cm:
                return "ICD9CM"
            elif code in icd10cm:
                return "ICD10CM"
            else:
                return "UNKNOWN"
        
        table = "diagnosis"
        # read table
        spark_df = spark.read.csv(os.path.join(self.root, f"{table}.csv"), header=True)
        if self.dev:
            spark_df = spark_df.limit(3000)
        
        spark_df = self.filter_events(
            spark=spark,
            spark_df=spark_df,
            timestamp_key="diagnosisoffset"
        )
        
        # sort by diagnosisoffset
        spark_df = spark_df.sort([self.visit_key, "diagnosisoffset"], ascending=True)
        
        schema = StructType(
            [
                StructField("patient_id", StringType(), False),
                StructField("visit_id", StringType(), False),
                StructField("code", ArrayType(StringType()), False),
                StructField("timestamp", ArrayType(StringType()), False),
                StructField("activeupondischarge", ArrayType(StringType()), False),
                StructField("diagnosisstring", ArrayType(StringType()), False),
                StructField("diagnosispriority", ArrayType(StringType()), False),
            ]
        )
        @F.pandas_udf(schema, functionType=F.PandasUDFType.GROUPED_MAP)
        def diagnosis_unit(df):
            visit_id = str(df[self.visit_key].iloc[0])
            patient_id = self.visit_id_to_patient_id[visit_id]
            code = sum(
                [
                    [c.strip() for c in codes.split(",")]
                    for codes in df["icd9code"].tolist() if codes is not None
                ], []
            )
            timestamp = [
                str(self.visit_id_to_encounter_time[visit_id] + pd.Timedelta(minutes=int(offset)))
                for offset in df["diagnosisoffset"].tolist()
            ]
            activeupondischarge = df["activeupondischarge"].tolist()
            diagnosisstring = df["diagnosisstring"].tolist()
            diagnosispriority = df["diagnosispriority"].tolist()
            return pd.DataFrame([[
                patient_id,
                visit_id,
                code,
                timestamp,
                activeupondischarge,
                diagnosisstring,
                diagnosispriority
            ]])
        
        pandas_df = spark_df.groupBy(self.visit_key).apply(diagnosis_unit).toPandas()
        
        def aggregate_events(row):
            events = []
            p_id = row["patient_id"]
            v_id = row["visit_id"]
            for code, timestamp, activeupondischarge, diagnosisstring, diagnosispriority in zip(
                row["code"], row["timestamp"], row["activeupondischarge"], row["diagnosisstring"],
                row["diagnosispriority"]
            ):
                vocab = icd9cm_or_icd10cm(code)
                event = Event(
                    code=code,
                    table=table,
                    vocabulary=vocab,
                    visit_id=v_id,
                    patient_id=p_id,
                    timestamp=strptime(timestamp),
                    activeupondischarge=activeupondischarge,
                    diagnosisstring=diagnosisstring,
                    diagnosispriority=diagnosispriority
                )
                events.append(event)
            return events
        
        # parallel apply to aggregate events
        aggregated_events = pandas_df.parallel_apply(aggregate_events, axis=1)
        
        # summarize the results
        patients = self._add_events_to_patient_dict(patients, aggregated_events)
        return patients

    def parse_treatment(
        self,
        patients: Dict[str, Patient],
        spark: SparkSession
    ) -> Dict[str, Patient]:
        """Helper function which parses treatment table.
        
        Will be called in `self.parse_tables()`.
        
        Docs:
            - treatment: https://eicu-crd.mit.edu/eicutables/treatment/
        
        Args:
            patients: a dict of Patient objects indexed by patient_id.
            spark: a spark session for reading the table.
        
        Returns:
            The updated patients dict.
        """
        table = "treatment"
        
        # read table
        spark_df = spark.read.csv(os.path.join(self.root, f"{table}.csv"), header=True)
        if self.dev:
            spark_df = spark_df.limit(3000)

        spark_df = self.filter_events(
            spark=spark,
            spark_df=spark_df,
            timestamp_key="treatmentoffset"
        )

        # sort by treatmentoffset
        spark_df = spark_df.sort([self.visit_key, "treatmentoffset"], ascending=True)
        
        schema = StructType(
            [
                StructField("patient_id", StringType(), False),
                StructField("visit_id", StringType(), False),
                StructField("code", ArrayType(StringType()), False),
                StructField("timestamp", ArrayType(StringType()), False),
                StructField("activeupondischarge", ArrayType(StringType()), False),
            ]
        )
        @F.pandas_udf(schema, functionType=F.PandasUDFType.GROUPED_MAP)
        def treatment_unit(df):
            visit_id = str(df[self.visit_key].iloc[0])
            patient_id = self.visit_id_to_patient_id[visit_id]
            code = df["treatmentstring"].tolist()
            timestamp = [
                str(self.visit_id_to_encounter_time[visit_id] + pd.Timedelta(minutes=int(offset)))
                for offset in df["treatmentoffset"].tolist()
            ]
            activeupondischarge = df["activeupondischarge"].tolist()
            return pd.DataFrame([[
                patient_id,
                visit_id,
                code,
                timestamp,
                activeupondischarge
            ]])
        
        pandas_df = spark_df.groupBy(self.visit_key).apply(treatment_unit).toPandas()
        
        def aggregate_events(row):
            events = []
            p_id = row["patient_id"]
            v_id = row["visit_id"]
            for code, timestamp, activeupondischarge in zip(
                row["code"], row["timestamp"], row["activeupondischarge"]
            ):
                event = Event(
                    code=code,
                    table=table,
                    vocabulary="eICU_TREATMENTSTRING",
                    visit_id=v_id,
                    patient_id=p_id,
                    timestamp=strptime(timestamp),
                    activeupondischarge=activeupondischarge
                )
                events.append(event)
            return events
        
        # parallel apply to aggregate events
        aggregated_events = pandas_df.parallel_apply(aggregate_events, axis=1)
        
        # summarize the results
        patients = self._add_events_to_patient_dict(patients, aggregated_events)
        return patients

    def parse_medication(
        self,
        patients: Dict[str, Patient],
        spark: SparkSession
    ) -> Dict[str, Patient]:
        """Helper function which parses medication table.
        
        Will be called in `self.parse_tables()`.
        
        Docs:
            - medication: https://eicu-crd.mit.edu/eicutables/medication/
        
        Args:
            patients: a dict of Patient objects indexed by patient_id.
            spark: a spark session for reading the table.
        
        Returns:
            The updated patients dict.
        """
        table = "medication"
        
        # read table
        spark_df = spark.read.csv(os.path.join(self.root, f"{table}.csv"), header=True)
        if self.dev:
            spark_df = spark_df.limit(3000)
        
        spark_df = self.filter_events(
            spark=spark,
            spark_df=spark_df,
            timestamp_key="drugstartoffset"
        )
        
        # sort by drugstartoffset
        spark_df = spark_df.sort([self.visit_key, "drugstartoffset"], ascending=True)
        
        schema = StructType(
            [
                StructField("patient_id", StringType(), False),
                StructField("visit_id", StringType(), False),
                StructField("code", ArrayType(StringType()), False),
                StructField("timestamp", ArrayType(StringType()), False),
                StructField("drugorderoffset", ArrayType(StringType()), False),
                StructField("drugivadmixture", ArrayType(StringType()), False),
                StructField("drugordercancelled", ArrayType(StringType()), False),
                StructField("drughiclseqno", ArrayType(StringType()), False),
                StructField("dosage", ArrayType(StringType()), False),
                StructField("routeadmin", ArrayType(StringType()), False),
                StructField("frequency", ArrayType(StringType()), False),
                StructField("loadingdose", ArrayType(StringType()), False),
                StructField("prn", ArrayType(StringType()), False),
                StructField("drugstopoffset", ArrayType(StringType()), False),
                StructField("gtc", ArrayType(StringType()), False)
            ]
        )
        @F.pandas_udf(schema, functionType=F.PandasUDFType.GROUPED_MAP)
        def medication_unit(df):
            visit_id = str(df[self.visit_key].iloc[0])
            patient_id = self.visit_id_to_patient_id[visit_id]
            code = df["drugname"].tolist()
            timestamp = [
                str(self.visit_id_to_encounter_time[visit_id] + pd.Timedelta(minutes=int(offset)))
                for offset in df["drugstartoffset"].tolist()
            ]
            drugorderoffset = df["drugorderoffset"].tolist()
            drugivadmixture = df["drugivadmixture"].tolist()
            drugordercancelled = df["drugordercancelled"].tolist()
            drughiclseqno = df["drughiclseqno"].tolist()
            dosage = df["dosage"].tolist()
            routeadmin = df["routeadmin"].tolist()
            frequency = df["frequency"].tolist()
            loadingdose = df["loadingdose"].tolist()
            prn = df["prn"].tolist()
            drugstopoffset = df["drugstopoffset"].tolist()
            gtc = df["gtc"].tolist()
            return pd.DataFrame([[
                patient_id,
                visit_id,
                code,
                timestamp,
                drugorderoffset,
                drugivadmixture,
                drugordercancelled,
                drughiclseqno,
                dosage,
                routeadmin,
                frequency,
                loadingdose,
                prn,
                drugstopoffset,
                gtc
            ]])

        pandas_df = spark_df.groupBy(self.visit_key).apply(medication_unit).toPandas()
        
        def aggregate_events(row):
            events = []
            p_id = row["patient_id"]
            v_id = row["visit_id"]
            for (
                code, timestamp, drugorderoffset, drugivadmixture, drugordercancelled,
                drughiclseqno, dosage, routeadmin, frequency, loadingdose, prn, drugstopoffset, gtc
            ) in zip(
                row["code"], row["timestamp"], row["drugorderoffset"], row["drugivadmixture"],
                row["drugordercancelled"], row["drughiclseqno"], row["dosage"], row["routeadmin"],
                row["frequency"], row["loadingdose"], row["prn"], row["drugstopoffset"], row["gtc"]
            ):
                event = Event(
                    code=code,
                    table=table,
                    vocabulary="eICU_DRUGNAME",
                    visit_id=v_id,
                    patient_id=p_id,
                    timestamp=strptime(timestamp),
                    drugorderoffset=drugorderoffset,
                    drugivadmixture=drugivadmixture,
                    drugordercancelled=drugordercancelled,
                    drughiclseqno=drughiclseqno,
                    dosage=dosage,
                    routeadmin=routeadmin,
                    frequency=frequency,
                    loadingdose=loadingdose,
                    prn=prn,
                    drugstopoffset=drugstopoffset,
                    gtc=gtc
                )
                events.append(event)
            return events
        
        # parallel apply to aggregate events
        aggregated_events = pandas_df.parallel_apply(aggregate_events, axis=1)
        
        # summarize the results
        patients = self._add_events_to_patient_dict(patients, aggregated_events)
        return patients

    def parse_lab(
        self,
        patients: Dict[str, Patient],
        spark: SparkSession
    ) -> Dict[str, Patient]:
        """Helper function which parses lab table.

        Will be called in `self.parse_tables()`.

        Docs:
            - lab: https://eicu-crd.mit.edu/eicutables/lab/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.
            spark: a spark session for reading the table.

        Returns:
            The updated patients dict.
        """
        table = "lab"
        
        # read table
        spark_df = spark.read.csv(os.path.join(self.root, f"{table}.csv"), header=True)
        if self.dev:
            spark_df = spark_df.limit(3000)

        spark_df = self.filter_events(
            spark=spark,
            spark_df=spark_df,
            timestamp_key="labresultoffset",
        )
        
        # sort by labresultoffset
        spark_df = spark_df.sort([self.visit_key, "labresultoffset"], ascending=True)
        
        schema = StructType(
            [
                StructField("patient_id", StringType(), False),
                StructField("visit_id", StringType(), False),
                StructField("code", ArrayType(StringType()), False),
                StructField("timestamp", ArrayType(StringType()), False),
                StructField("labtypeid", ArrayType(StringType()), False),
                StructField("labresult", ArrayType(StringType()), False),
                StructField("labresulttext", ArrayType(StringType()), False),
                StructField("labmeasurenamesystem", ArrayType(StringType()), False),
                StructField("labmeasurenameinterface", ArrayType(StringType()), False),
                StructField("labresultrevisedoffset", ArrayType(StringType()), False)
            ]
        )
        @F.pandas_udf(schema, functionType=F.PandasUDFType.GROUPED_MAP)
        def lab_unit(df):
            visit_id = str(df[self.visit_key].iloc[0])
            patient_id = self.visit_id_to_patient_id[visit_id]
            code = df["labname"].tolist()
            timestamp = [
                str(self.visit_id_to_encounter_time[visit_id] + pd.Timedelta(minutes=int(offset)))
                for offset in df["labresultoffset"].tolist()
            ]
            labtypeid = df["labtypeid"].tolist()
            labresult = df["labresult"].tolist()
            labresulttext = df["labresulttext"].tolist()
            labmeasurenamesystem = df["labmeasurenamesystem"].tolist()
            labmeasurenameinterface = df["labmeasurenameinterface"].tolist()
            labresultrevisedoffset = df["labresultrevisedoffset"].tolist()
            return pd.DataFrame([[
                patient_id,
                visit_id,
                code,
                timestamp,
                labtypeid,
                labresult,
                labresulttext,
                labmeasurenamesystem,
                labmeasurenameinterface,
                labresultrevisedoffset
            ]])

        pandas_df = spark_df.groupBy(self.visit_key).apply(lab_unit).toPandas()
        
        def aggregate_events(row):
            events = []
            p_id = row["patient_id"]
            v_id = row["visit_id"]
            for (
                code, timestamp, labtypeid, labresult, labresulttext,
                labmeasurenamesystem, labmeasurenameinterface, labresultrevisedoffset
            ) in zip(
                row["code"], row["timestamp"], row["labtypeid"], row["labresult"],
                row["labresulttext"], row["labmeasurenamesystem"], row["labmeasurenameinterface"],
                row["labresultrevisedoffset"]
            ):
                event = Event(
                    code=code,
                    table=table,
                    vocabulary="eICU_LABNAME",
                    visit_id=v_id,
                    patient_id=p_id,
                    timestamp=strptime(timestamp),
                    labtypeid=labtypeid,
                    labresult=labresult,
                    labresulttext=labresulttext,
                    labmeasurenamesystem=labmeasurenamesystem,
                    labmeasurenameinterface=labmeasurenameinterface,
                    labresultrevisedoffset=labresultrevisedoffset
                )
                events.append(event)
            return events

        # parallel apply to aggregate events
        aggregated_events = pandas_df.parallel_apply(aggregate_events, axis=1)
        
        # summarize the results
        patients = self._add_events_to_patient_dict(patients, aggregated_events)
        return patients

    def parse_physicalexam(
        self,
        patients: Dict[str, Patient],
        spark: SparkSession
    ) -> Dict[str, Patient]:
        """Helper function which parses physicalExam table
        
        Will be called in `self.parse_tables()`.
        
        Docs:
            - physicalExam: https://eicu-crd.mit.edu/eicutables/physicalexam/
        
        Args:
            patients: a dict of `Patient` objects indexed by patient_id.
            spark: a spark session for reading the table.
        
        Returns:
            The updated patients dict.
        """
        table = "physicalExam"
        
        # read table
        spark_df = spark.read.csv(os.path.join(self.root, f"{table}.csv"), header=True)
        if self.dev:
            spark_df = spark_df.limit(3000)
        
        spark_df = self.filter_events(
            spark=spark,
            spark_df=spark_df,
            timestamp_key="physicalexamoffset"
        )
        
        # sort by physicalexamoffset
        spark_df = spark_df.sort([self.visit_key, "physicalexamoffset"], ascending=True)
        
        schema = StructType(
            [
                StructField("patient_id", StringType(), False),
                StructField("visit_id", StringType(), False),
                StructField("code", ArrayType(StringType()), False),
                StructField("timestamp", ArrayType(StringType()), False),
                StructField("physicalexamvalue", ArrayType(StringType()), False),
                StructField("physicalexamtext", ArrayType(StringType()), False),
            ]
        )
        @F.pandas_udf(schema, functionType=F.PandasUDFType.GROUPED_MAP)
        def physicalexam_unit(df):
            visit_id = str(df[self.visit_key].iloc[0])
            patient_id = self.visit_id_to_patient_id[visit_id]
            code = df["physicalexampath"].tolist()
            timestamp = [
                str(self.visit_id_to_encounter_time[visit_id] + pd.Timedelta(minutes=int(offset)))
                for offset in df["physicalexamoffset"].tolist()
            ]
            physicalexamvalue = df["physicalexamvalue"].tolist()
            physicalexamtext = df["physicalexamtext"].tolist()
            return pd.DataFrame([[
                patient_id,
                visit_id,
                code,
                timestamp,
                physicalexamvalue,
                physicalexamtext
            ]])
        
        pandas_df = spark_df.groupBy(self.visit_key).apply(physicalexam_unit).toPandas()
        
        def aggregate_events(row):
            events = []
            p_id = row["patient_id"]
            v_id = row["visit_id"]
            for code, timestamp, physicalexamvalue, physicalexamtext in zip(
                row["code"], row["timestamp"], row["physicalexamvalue"], row["physicalexamtext"]
            ):
                event = Event(
                    code=code,
                    table=table,
                    vocabulary="eICU_PHYSICALEXAMPATH",
                    visit_id=v_id,
                    patient_id=p_id,
                    timestamp=strptime(timestamp),
                    physicalexamvalue=physicalexamvalue,
                    physicalexamtext=physicalexamtext
                )
                events.append(event)
            return events

        # parallel apply to aggregate events
        aggregated_events = pandas_df.parallel_apply(aggregate_events, axis=1)
        
        # summarize the results
        patients = self._add_events_to_patient_dict(patients, aggregated_events)
        return patients

    def parse_admissiondx(
        self,
        patients: Dict[str, Patient],
        spark: SparkSession
    ) -> Dict[str, Patient]:
        """Helper function which parses admissionDx (admission diagnosis) table.

        Will be called in `self.parse_tables()`.

        Docs:
            - admissionDx: https://eicu-crd.mit.edu/eicutables/admissiondx/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.
            spark: a spark session for reading the table.

        Returns:
            The updated patients dict.
        """
        table = "admissionDx"
        
        # read table
        spark_df = spark.read.csv(os.path.join(self.root, f"{table}.csv"), header=True)
        if self.dev:
            spark_df = spark_df.limit(3000)
        
        spark_df = self.filter_events(
            spark=spark,
            spark_df=spark_df,
            timestamp_key="admitdxenteredoffset"
        )
        
        # sort by admitdxenteredoffset
        spark_df = spark_df.sort([self.visit_key, "admitdxenteredoffset"], ascending=True)
        
        schema = StructType(
            [
                StructField("patient_id", StringType(), False),
                StructField("visit_id", StringType(), False),
                StructField("code", ArrayType(StringType()), False),
                StructField("timestamp", ArrayType(StringType()), False),
                StructField("admitdxname", ArrayType(StringType()), False),
                StructField("admitdxtext", ArrayType(StringType()), False),
            ]
        )
        @F.pandas_udf(schema, functionType=F.PandasUDFType.GROUPED_MAP)
        def admissionDx_unit(df):
            visit_id = str(df[self.visit_key].iloc[0])
            patient_id = self.visit_id_to_patient_id[visit_id]
            code = df["admitdxpath"].tolist()
            timestamp = [
                str(self.visit_id_to_encounter_time[visit_id] + pd.Timedelta(minutes=int(offset)))
                for offset in df["admitdxenteredoffset"].tolist()
            ]
            admitdxname = df["admitdxname"].tolist()
            admitdxtext = df["admitdxtext"].tolist()
            return pd.DataFrame([[
                patient_id,
                visit_id,
                code,
                timestamp,
                admitdxname,
                admitdxtext
            ]])
        
        pandas_df = spark_df.groupBy(self.visit_key).apply(admissionDx_unit).toPandas()
        
        def aggregate_events(row):
            events = []
            p_id = row["patient_id"]
            v_id = row["visit_id"]
            for code, timestamp, admitdxname, admitdxtext in zip(
                row["code"], row["timestamp"], row["admitdxname"], row["admitdxtext"]
            ):
                event = Event(
                    code=code,
                    table=table,
                    vocabulary="eicu_ADMITDXPATH",
                    visit_id=v_id,
                    patient_id=p_id,
                    timestamp=strptime(timestamp),
                    admitdxname=admitdxname,
                    admitdxtext=admitdxtext
                )
                events.append(event)
            return events
        
        # parallel apply to aggregate events
        aggregated_events = pandas_df.parallel_apply(aggregate_events, axis=1)
        
        # summarize the results
        patients = self._add_events_to_patient_dict(patients, aggregated_events)
        return patients

    def parse_infusiondrug(
        self,
        patients: Dict[str, Patient],
        spark: SparkSession
    ) -> Dict[str, Patient]:
        """Helper function which parses infusionDrug table
        
        Will be called in `self.parse_tables()`.
        
        Docs:
            - infusionDrug: https://eicu-crd.mit.edu/eicutables/infusiondrug/
        
        Args:
            patients: a dict of `Patient` objects indexed by patient_id.
            spark: a spark session for reading the table.
        
        Returns:
            The updated patients dict.
        """
        table = "infusionDrug"
        
        # read table
        spark_df = spark.read.csv(os.path.join(self.root, f"{table}.csv"), header=True)
        if self.dev:
            spark_df = spark_df.limit(3000)

        spark_df = self.filter_events(
            spark=spark,
            spark_df=spark_df,
            timestamp_key="infusionoffset"
        )

        # sort by infusionoffset
        spark_df = spark_df.sort([self.visit_key, "infusionoffset"], ascending=True)
        
        schema = StructType(
            [
                StructField("patient_id", StringType(), False),
                StructField("visit_id", StringType(), False),
                StructField("code", ArrayType(StringType()), False),
                StructField("timestamp", ArrayType(StringType()), False),
                StructField("drugrate", ArrayType(StringType()), False),
                StructField("infusionrate", ArrayType(StringType()), False),
                StructField("drugamount", ArrayType(StringType()), False),
                StructField("volumeoffluid", ArrayType(StringType()), False),
                StructField("patientweight", ArrayType(StringType()), False)
            ]
        )
        @F.pandas_udf(schema, functionType=F.PandasUDFType.GROUPED_MAP)
        def infusionDrug_unit(df):
            visit_id = str(df[self.visit_key].iloc[0])
            patient_id = self.visit_id_to_patient_id[visit_id]
            code = df["drugname"].tolist()
            timestamp = [
                str(self.visit_id_to_encounter_time[visit_id] + pd.Timedelta(minutes=int(offset)))
                for offset in df["infusionoffset"].tolist()
            ]
            drugrate = df["drugrate"].tolist()
            infusionrate = df["infusionrate"].tolist()
            drugamount = df["drugamount"].tolist()
            volumeoffluid = df["volumeoffluid"].tolist()
            patientweight = df["patientweight"].tolist()
            return pd.DataFrame([[
                patient_id,
                visit_id,
                code,
                timestamp,
                drugrate,
                infusionrate,
                drugamount,
                volumeoffluid,
                patientweight
            ]])

        pandas_df = spark_df.groupBy(self.visit_key).apply(infusionDrug_unit).toPandas()
        
        def aggregate_events(row):
            events = []
            p_id = row["patient_id"]
            v_id = row["visit_id"]
            for (
                code, timestamp, drugrate, infusionrate, drugamount, volumeoffluid, patientweight
            ) in zip(
                row["code"], row["timestamp"], row["drugrate"], row["infusionrate"],
                row["drugamount"], row["volumeoffluid"], row["patientweight"]
            ):
                event = Event(
                    code=code,
                    table=table,
                    vocabulary="eICU_DRUGNAME",
                    visit_id=v_id,
                    patient_id=p_id,
                    timestamp=strptime(timestamp),
                    drugrate=drugrate,
                    infusionrate=infusionrate,
                    drugamount=drugamount,
                    volumeoffluid=volumeoffluid,
                    patientweight=patientweight
                )
                events.append(event)
            return events

        # parallel apply to aggregate events
        aggregated_events = pandas_df.parallel_apply(aggregate_events, axis=1)
        
        # summarize the results
        patients = self._add_events_to_patient_dict(patients, aggregated_events)
        return patients

    def parse_nursecare(
        self,
        patients: Dict[str, Patient],
        spark: SparkSession
    ) -> Dict[str, Patient]:
        """Helper function which parses nurseCare table
        
        Will be called in `self.parse_tables()`.
        
        Docs:
            - nurseCare: https://eicu-crd.mit.edu/eicutables/nursecare/
        
        Args:
            patients: a dict of `Patient` objects indexed by patient_id.
            spark: a spark session for reading the table.
        
        Returns:
            The updated patients dict.
        """
        table = "nurseCare"
        
        # read table
        spark_df = spark.read.csv(os.path.join(self.root, f"{table}.csv"), header=True)
        if self.dev:
            spark_df = spark_df.limit(3000)
        
        spark_df = self.filter_events(
            spark=spark,
            spark_df=spark_df,
            timestamp_key="nursecareoffset"
        )
        
        # sort by nursecareoffset
        spark_df = spark_df.sort([self.visit_key, "nursecareoffset"], ascending=True)
        
        schema = StructType(
            [
                StructField("patient_id", StringType(), False),
                StructField("visit_id", StringType(), False),
                StructField("code", ArrayType(StringType()), False),
                StructField("timestamp", ArrayType(StringType()), False),
                StructField("celllabel", ArrayType(StringType()), False),
                StructField("cellattribute", ArrayType(StringType()), False),
                StructField("cellattributevalue", ArrayType(StringType()), False),
            ]
        )
        @F.pandas_udf(schema, functionType=F.PandasUDFType.GROUPED_MAP)
        def nurseCare_unit(df):
            visit_id = str(df[self.visit_key].iloc[0])
            patient_id = self.visit_id_to_patient_id[visit_id]
            code = df["cellattributepath"].tolist()
            timestamp = [
                str(self.visit_id_to_encounter_time[visit_id] + pd.Timedelta(minutes=int(offset)))
                for offset in df["nursecareoffset"].tolist()
            ]
            celllabel = df["celllabel"].tolist()
            cellattribute = df["cellattribute"].tolist()
            cellattributevalue = df["cellattributevalue"].tolist()
            return pd.DataFrame([[
                patient_id,
                visit_id,
                code,
                timestamp,
                celllabel,
                cellattribute,
                cellattributevalue
            ]])
        
        pandas_df = spark_df.groupBy(self.visit_key).apply(nurseCare_unit).toPandas()
        
        def aggregate_events(row):
            events = []
            p_id = row["patient_id"]
            v_id = row["visit_id"]
            for code, timestamp, celllabel, cellattribute, cellattributevalue in zip(
                row["code"], row["timestamp"], row["celllabel"], row["cellattribute"],
                row["cellattributevalue"]
            ):
                event = Event(
                    code=code,
                    table=table,
                    vocabulary="eicu_CELLATTRIBUTEPATH",
                    visit_id=v_id,
                    patient_id=p_id,
                    timestamp=strptime(timestamp),
                    celllabel=celllabel,
                    cellattribute=cellattribute,
                    cellattributevalue=cellattributevalue
                )
                events.append(event)
            return events
        
        # parallel apply to aggregate events
        aggregated_events = pandas_df.parallel_apply(aggregate_events, axis=1)
        
        # summarize the results
        patients = self._add_events_to_patient_dict(patients, aggregated_events)
        return patients

    def parse_nursecharting(
        self,
        patients: Dict[str, Patient],
        spark: SparkSession
    ) -> Dict[str, Patient]:
        """Helper function which parses nurseCharting table.

        Will be called in `self.parse_tables()`.

        Docs:
            - nurseCharting: https://eicu-crd.mit.edu/eicutables/nursecharting/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.
            spark: a spark session for reading the table.

        Returns:
            The updated patients dict.
        """
        table = "nurseCharting"
        
        # read table
        spark_df = spark.read.csv(os.path.join(self.root, f"{table}.csv"), header=True)
        if self.dev:
            spark_df = spark_df.limit(3000)
        
        spark_df = self.filter_events(
            spark=spark,
            spark_df=spark_df,
            timestamp_key="nursingchartoffset"
        )
        
        # sort by nursingchartoffset
        spark_df = spark_df.sort([self.visit_key, "nursingchartoffset"], ascending=True)
        
        schema = StructType(
            [
                StructField("patient_id", StringType(), False),
                StructField("visit_id", StringType(), False),
                StructField("code", ArrayType(StringType()), False),
                StructField("timestamp", ArrayType(StringType()), False),
                StructField("nursingchartcelltypecat", ArrayType(StringType()), False),
                StructField("nursingchartcelltypevallabel", ArrayType(StringType()), False),
                StructField("nursingchartvalue", ArrayType(StringType()), False),
            ]
        )
        @F.pandas_udf(schema, functionType=F.PandasUDFType.GROUPED_MAP)
        def nurseCharting_unit(df):
            visit_id = str(df[self.visit_key].iloc[0])
            patient_id = self.visit_id_to_patient_id[visit_id]
            code = df["nursingchartcelltypevalname"].tolist()
            timestamp = [
                str(self.visit_id_to_encounter_time[visit_id] + pd.Timedelta(minutes=int(offset)))
                for offset in df["nursingchartoffset"].tolist()
            ]
            nursingchartcelltypecat = df["nursingchartcelltypecat"].tolist()
            nursingchartcelltypevallabel = df["nursingchartcelltypevallabel"].tolist()
            nursingchartvalue = df["nursingchartvalue"]
            return pd.DataFrame([[
                patient_id,
                visit_id,
                code,
                timestamp,
                nursingchartcelltypecat,
                nursingchartcelltypevallabel,
                nursingchartvalue
            ]])
        
        pandas_df = spark_df.groupBy(self.visit_key).apply(nurseCharting_unit).toPandas()
        
        def aggregate_events(row):
            events = []
            p_id = row["patient_id"]
            v_id = row["visit_id"]
            for (
                code, timestamp, nursingchartcelltypecat, nursingchartcelltypevallabel,
                nursingchartvalue
            ) in zip(
                row["code"], row["timestamp"], row["nursingchartcelltypecat"],
                row["nursingchartcelltypevallabel"], row["nursingchartvalue"]
            ):
                event = Event(
                    code=code,
                    table=table,
                    vocabulary="eicu_NURSINGCHARTCELLTYPEVALNAME",
                    visit_id=v_id,
                    patient_id=p_id,
                    timestamp=strptime(timestamp),
                    nursingchartcelltypecat=nursingchartcelltypecat,
                    nursingchartcelltypevallabel=nursingchartcelltypevallabel,
                    nursingchartvalue=nursingchartvalue
                )
                events.append(event)
            return events
        
        # parallel apply to aggregate events
        aggregated_events = pandas_df.parallel_apply(aggregate_events, axis=1)

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, aggregated_events)
        return patients

    def parse_intakeoutput(
        self,
        patients: Dict[str, Patient],
        spark: SparkSession
    ) -> Dict[str, Patient]:
        """Helper function which parses intakeOuptut table

        Will be called in `self.parse_tables()`.
        
        Docs:
            - intakeOutput: https://eicu-crd.mit.edu/eicutables/intakeoutput/
        
        Args:
            patients: a dict of `Patient` objects indexed by patient_id.
            spark: a spark session for reading the table.
        
        Returns:
            The updated patients dict.
        """
        table = "intakeOutput"
        
        # read table
        spark_df = spark.read.csv(os.path.join(self.root, f"{table}.csv"), header=True)
        if self.dev:
            spark_df = spark_df.limit(3000)
        
        spark_df = self.filter_events(
            spark=spark,
            spark_df=spark_df,
            timestamp_key = "intakeoutputoffset"
        )
        
        # sort by intakeoutputoffset
        spark_df = spark_df.sort([self.visit_key, "intakeoutputoffset"], ascending=True)
        
        schema = StructType(
            [
                StructField("patient_id", StringType(), False),
                StructField("visit_id", StringType(), False),
                StructField("code", ArrayType(StringType()), False),
                StructField("timestamp", ArrayType(StringType()), False),
                StructField("intaketotal", ArrayType(StringType()), False),
                StructField("outputtotal", ArrayType(StringType()), False),
                StructField("dialysistotal", ArrayType(StringType()), False),
                StructField("nettotal", ArrayType(StringType()), False),
                StructField("celllabel", ArrayType(StringType()), False),
                StructField("cellvaluenumeric", ArrayType(StringType()), False),
                StructField("cellvaluetext", ArrayType(StringType()), False),
            ]
        )
        @F.pandas_udf(schema, functionType=F.PandasUDFType.GROUPED_MAP)
        def intakeOutput_unit(df):
            visit_id = str(df[self.visit_key].iloc[0])
            patient_id = self.visit_id_to_patient_id[visit_id]
            code = df["cellpath"]
            timestamp = [
                str(self.visit_id_to_encounter_time[visit_id] + pd.Timedelta(minutes=int(offset)))
                for offset in df["intakeoutputoffset"].tolist()
            ]
            intaketotal = df["intaketotal"].tolist()
            outputtotal = df["outputtotal"].tolist()
            dialysistotal = df["dialysistotal"].tolist()
            nettotal = df["nettotal"].tolist()
            celllabel = df["celllabel"].tolist()
            cellvaluenumeric = df["cellvaluenumeric"].tolist()
            cellvaluetext = df["cellvaluetext"].tolist()
            return pd.DataFrame([[
                patient_id,
                visit_id,
                code,
                timestamp,
                intaketotal,
                outputtotal,
                dialysistotal,
                nettotal,
                celllabel,
                cellvaluenumeric,
                cellvaluetext
            ]])
        
        pandas_df = spark_df.groupBy(self.visit_key).apply(intakeOutput_unit).toPandas()
        
        def aggregate_events(row):
            events = []
            p_id = row["patient_id"]
            v_id = row["visit_id"]
            for (
                code, timestamp, intaketotal, outputtotal, dialysistotal, nettotal, celllabel,
                cellvaluenumeric, cellvaluetext
            ) in zip(
                row["code"], row["timestamp"], row["intaketotal"], row["outputtotal"],
                row["dialysistotal"], row["nettotal"], row["celllabel"], row["cellvaluenumeric"],
                row["cellvaluetext"]
            ):
                event = Event(
                    code=code,
                    table=table,
                    vocabulary="eicu_INTAKEOUTPUTCELLPATH",
                    visit_id=v_id,
                    patient_id=p_id,
                    timestamp=strptime(timestamp),
                    intaketotal=intaketotal,
                    outputtotal=outputtotal,
                    dialysistotal=dialysistotal,
                    nettotal=nettotal,
                    celllabel=celllabel,
                    cellvaluenumeric=cellvaluenumeric,
                    cellvaluetext=cellvaluetext
                )
                events.append(event)
            return events
        
        # parallel apply to aggregate events
        aggregated_events = pandas_df.parallel_apply(aggregate_events, axis=1)

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, aggregated_events)
        return patients
    
    def parse_microlab(
        self,
        patients: Dict[str, Patient],
        spark: SparkSession
    ) -> Dict[str, Patient]:
        """Helper function which parses microLab table
        
        Will be called in `self.parse_tables()`.
        
        Docs:
            - microLab: https://eicu-crd.mit.edu/eicutables/microlab/
        
        Args:
            patients: a dict of `Patient` objects indexed by patient_id.
            spark: a spark session for reading the table.
        
        Returns:
            The updated patients dict.
        
        Notes:
            `code` is set to `None` for all events in this table because there is no specific
            column for `code` in microLab table.
        """
        table = "microLab"
        
        # read table
        spark_df = spark.read.csv(os.path.join(self.root, f"{table}.csv"), header=True)
        if self.dev:
            spark_df = spark_df.limit(3000)
        
        spark_df = self.filter_events(
            spark=spark,
            spark_df=spark_df,
            timestamp_key="culturetakenoffset"
        )
        
        # sort by culturetakenoffset
        spark_df = spark_df.sort([self.visit_key, "culturetakenoffset"], ascending=True)
        
        schema = StructType(
            [
                StructField("patient_id", StringType(), False),
                StructField("visit_id", StringType(), False),
                StructField("code", ArrayType(StringType()), False),
                StructField("timestamp", ArrayType(StringType()), False),
                StructField("culturesite", ArrayType(StringType()), False),
                StructField("organism", ArrayType(StringType()), False),
                StructField("antibiotic", ArrayType(StringType()), False),
                StructField("sensitivitylevel", ArrayType(StringType()), False),
            ]
        )
        @F.pandas_udf(schema, functionType=F.PandasUDFType.GROUPED_MAP)
        def microLab_unit(df):
            visit_id = str(df[self.visit_key].iloc[0])
            patient_id = self.visit_id_to_patient_id[visit_id]
            code = [None] * len(df)
            timestamp = [
                str(self.visit_id_to_encounter_time[visit_id] + pd.Timedelta(minutes=int(offset)))
                for offset in df["culturetakenoffset"].tolist()
            ]
            culturesite = df["culturesite"].tolist()
            organism = df["organism"].tolist()
            antibiotic = df["antibiotic"].tolist()
            sensitivitylevel = df["sensitivitylevel"].tolist()
            return pd.DataFrame([[
                patient_id,
                visit_id,
                code,
                timestamp,
                culturesite,
                organism,
                antibiotic,
                sensitivitylevel
            ]])
        
        pandas_df = spark_df.groupBy(self.visit_key).apply(microLab_unit).toPandas()
        
        def aggregate_events(row):
            events = []
            p_id = row["patient_id"]
            v_id = row["visit_id"]
            for code, timestamp, culturesite, organism, antibiotic, sensitivitylevel in zip(
                row["code"], row["timestamp"], row["culturesite"], row["organism"],
                row["antibiotic"], row["sensitivitylevel"]
            ):
                event = Event(
                    code=code,
                    table=table,
                    vocabulary="eICU_MICROLABSTRING",
                    visit_id=v_id,
                    patient_id=p_id,
                    timestamp=strptime(timestamp),
                    culturesite=culturesite,
                    organism=organism,
                    antibiotic=antibiotic,
                    sensitivitylevel=sensitivitylevel
                )
                events.append(event)
            return events

        # parallel apply to aggregate events
        aggregated_events = pandas_df.parallel_apply(aggregate_events, axis=1)
        
        # summarize the results
        patients = self._add_events_to_patient_dict(patients, aggregated_events)
        return patients

    def parse_nurseassessment(
        self,
        patients: Dict[str, Patient],
        spark: SparkSession
    ) -> Dict[str, Patient]:
        """Helper function which parses nurseAssessment table
        
        Will be called in `self.parse_tables()`.
        
        Docs:
            - nurseAssessment: https://eicu-crd.mit.edu/eicutables/nurseassessment/
        
        Args:
            patients: a dict of `Patient` objects indexed by patient_id.
            spark: a spark session for reading the table.
        
        Returns:
            The updated patients dict.
        """
        table = "nurseAssessment"
        
        # read table
        spark_df = spark.read.csv(os.path.join(self.root, f"{table}.csv"), header=True)
        if self.dev:
            spark_df = spark_df.limit(3000)
        
        spark_df = self.filter_events(
            spark=spark,
            spark_df=spark_df,
            timestamp_key="nurseassessoffset"
        )
        
        # sort by nurseassessoffset
        spark_df = spark_df.sort([self.visit_key, "nurseassessoffset"], ascending=True)
        
        schema = StructType(
            [
                StructField("patient_id", StringType(), False),
                StructField("visit_id", StringType(), False),
                StructField("code", ArrayType(StringType()), False),
                StructField("timestamp", ArrayType(StringType()), False),
                StructField("celllabel", ArrayType(StringType()), False),
                StructField("cellattribute", ArrayType(StringType()), False),
                StructField("cellattributevalue", ArrayType(StringType()), False),
            ]
        )
        @F.pandas_udf(schema, functionType=F.PandasUDFType.GROUPED_MAP)
        def nurseAssessment_unit(df):
            visit_id = str(df[self.visit_key].iloc[0])
            patient_id = self.visit_id_to_patient_id[visit_id]
            code = df["cellattributepath"].tolist()
            timestamp = [
                str(self.visit_id_to_encounter_time[visit_id] + pd.Timedelta(minutes=int(offset)))
                for offset in df["nurseassessoffset"].tolist()
            ]
            celllabel = df["celllabel"].tolist()
            cellattribute = df["cellattribute"].tolist()
            cellattributevalue = df["cellattributevalue"].tolist()
            return pd.DataFrame([[
                patient_id,
                visit_id,
                code,
                timestamp,
                celllabel,
                cellattribute,
                cellattributevalue
            ]])
        
        pandas_df = spark_df.groupBy(self.visit_key).apply(nurseAssessment_unit).toPandas()

        def aggregate_events(row):
            events = []
            p_id = row["patient_id"]
            v_id = row["visit_id"]
            for code, timestamp, celllabel, cellattribute, cellattributevalue in zip(
                row["code"], row["timestamp"], row["celllabel"], row["cellattribute"],
                row["cellattributevalue"]
            ):
                event = Event(
                    code=code,
                    table=table,
                    vocabulary="eicu_CELLATTRIBUTEPATH",
                    visit_id=v_id,
                    patient_id=p_id,
                    timestamp=strptime(timestamp),
                    celllabel=celllabel,
                    cellattribute=cellattribute,
                    cellattributevalue=cellattributevalue
                )
                events.append(event)
            return events
        
        # parallel apply to aggregate events
        aggregated_events = pandas_df.parallel_apply(aggregate_events, axis=1)
        
        # summarize the results
        patients = self._add_events_to_patient_dict(patients, aggregated_events)
        return patients

    def parse_vitalperiodic(
        self,
        patients: Dict[str, Patient],
        spark: SparkSession
    ) -> Dict[str, Patient]:
        """Helper function which parses vitalPeriodic table.

        Will be called in `self.parse_tables()`.

        Docs:
            - vitalPeriodic: https://eicu-crd.mit.edu/eicutables/vitalperiodic/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.
            spark: a spark session for reading the table.

        Returns:
            The updated patients dict.

        Notes:
            `code` is set to `None` for all events in this table because there is no specific
            column for `code` in vitalPeriodic table.
        """
        table = "vitalPeriodic"
        
        # read table
        spark_df = spark.read.csv(os.path.join(self.root, f"{table}.csv"), header=True)
        if self.dev:
            spark_df = spark_df.limit(3000)
        
        spark_df = self.filter_events(
            spark=spark,
            spark_df=spark_df,
            timestamp_key="observationoffset"
        )
        
        # sort by observationoffset
        spark_df = spark_df.sort([self.visit_key, "observationoffset"], ascending=True)
        
        schema = StructType(
            [
                StructField("patient_id", StringType(), False),
                StructField("visit_id", StringType(), False),
                StructField("code", ArrayType(StringType()), False),
                StructField("timestamp", ArrayType(StringType()), False),
                StructField("temperature", ArrayType(StringType()), False),
                StructField("sao2", ArrayType(StringType()), False),
                StructField("heartrate", ArrayType(StringType()), False),
                StructField("respiration", ArrayType(StringType()), False),
                StructField("cvp", ArrayType(StringType()), False),
                StructField("etco2", ArrayType(StringType()), False),
                StructField("systemicsystolic", ArrayType(StringType()), False),
                StructField("systemicdiastolic", ArrayType(StringType()), False),
                StructField("systemicmean", ArrayType(StringType()), False),
                StructField("pasystolic", ArrayType(StringType()), False),
                StructField("padiastolic", ArrayType(StringType()), False),
                StructField("pamean", ArrayType(StringType()), False),
                StructField("st1", ArrayType(StringType()), False),
                StructField("st2", ArrayType(StringType()), False),
                StructField("st3", ArrayType(StringType()), False),
                StructField("icp", ArrayType(StringType()), False),
            ]
        )
        @F.pandas_udf(schema, functionType=F.PandasUDFType.GROUPED_MAP)
        def vitalPeriodic_unit(df):
            visit_id = str(df[self.visit_key].iloc[0])
            patient_id = self.visit_id_to_patient_id[visit_id]
            code = [None] * len(df)
            timestamp = [
                str(self.visit_id_to_encounter_time[visit_id] + pd.Timedelta(minutes=int(offset)))
                for offset in df["observationoffset"].tolist()
            ]
            temperature = df["temperature"].tolist()
            sao2 = df["sao2"].tolist()
            heartrate = df["heartrate"].tolist()
            respiration = df["respiration"].tolist()
            cvp = df["cvp"].tolist()
            etco2 = df["etco2"].tolist()
            systemicsystolic = df["systemicsystolic"].tolist()
            systemicdiastolic = df["systemicdiastolic"].tolist()
            systemicmean = df["systemicmean"].tolist()
            pasystolic = df["pasystolic"].tolist()
            padiastolic = df["padiastolic"].tolist()
            pamean = df["pamean"].tolist()
            st1 = df["st1"].tolist()
            st2 = df["st2"].tolist()
            st3 = df["st3"].tolist()
            icp = df["icp"].tolist()
            return pd.DataFrame([[
                patient_id,
                visit_id,
                code,
                timestamp,
                temperature,
                sao2,
                heartrate,
                respiration,
                cvp,
                etco2,
                systemicsystolic,
                systemicdiastolic,
                systemicmean,
                pasystolic,
                padiastolic,
                pamean,
                st1,
                st2,
                st3,
                icp
            ]])
        
        pandas_df = spark_df.groupBy(self.visit_key).apply(vitalPeriodic_unit).toPandas()
        
        def aggregate_events(row):
            events = []
            p_id = row["patient_id"]
            v_id = row["visit_id"]
            for (
                code, timestamp, temperature, sao2, heartrate, respiration, cvp, etco2,
                systemicsystloic, systemicdiastolic, systemicmean, pasystolic, padiastolic,
                pamean, st1, st2, st3, icp
            ) in zip(
                row["code"], row["timestamp"], row["temperature"], row["sao2"], row["heartrate"],
                row["respiration"], row["cvp"], row["etco2"], row["systemicsystolic"],
                row["systemicdiastolic"], row["systemicmean"], row["pasystolic"],
                row["padiastolic"], row["pamean"], row["st1"], row["st2"], row["st3"], row["icp"]
            ):
                event = Event(
                    code=code,
                    table=table,
                    vocabulary="eicu_VITALPERIODICSTRING",
                    visit_id=v_id,
                    patient_id=p_id,
                    timestamp=strptime(timestamp),
                    temperature=temperature,
                    sao2=sao2,
                    heartrate=heartrate,
                    respiration=respiration,
                    cvp=cvp,
                    etco2=etco2,
                    systemicsystolic=systemicsystloic,
                    systemicdiastolic=systemicdiastolic,
                    systemicmean=systemicmean,
                    pasystolic=pasystolic,
                    padiastolic=padiastolic,
                    pamean=pamean,
                    st1=st1,
                    st2=st2,
                    st3=st3,
                    icp=icp
                )
                events.append(event)
            return events
        
        # parallel apply to aggregate events
        aggregated_events = pandas_df.parallel_apply(aggregate_events, axis=1)
        
        # summarize the results
        patients = self._add_events_to_patient_dict(patients, aggregated_events)
        return patients
    
    def parse_vitalaperiodic(
        self,
        patients: Dict[str, Patient],
        spark: SparkSession
    ) -> Dict[str, Patient]:
        """Helper function which parses vitalAperiodic table.

        Will be called in `self.parse_tables()`.

        Docs:
            - vitalAperiodic: https://eicu-crd.mit.edu/eicutables/vitalaperiodic/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.
            spark: a spark session for reading the table.

        Returns:
            The updated patients dict.

        Notes:
            `code` is set to `None` for all events in this table because there is no specific
            column for `code` in vitalAperiodic table.
        """
        table = "vitalAperiodic"
        
        # read table
        spark_df = spark.read.csv(os.path.join(self.root, f"{table}.csv"), header=True)
        if self.dev:
            spark_df = spark_df.limit(3000)
        
        spark_df = self.filter_events(
            spark=spark,
            spark_df=spark_df,
            timestamp_key="observationoffset"
        )
        
        # sort by observationoffset
        spark_df = spark_df.sort([self.visit_key, "observationoffset"], ascending=True)
        
        schema = StructType(
            [
                StructField("patient_id", StringType(), False),
                StructField("visit_id", StringType(), False),
                StructField("code", ArrayType(StringType()), False),
                StructField("timestamp", ArrayType(StringType()), False),
                StructField("noninvasivesystolic", ArrayType(StringType()), False),
                StructField("noninvasivediastolic", ArrayType(StringType()), False),
                StructField("noninvasivemean", ArrayType(StringType()), False),
                StructField("paop", ArrayType(StringType()), False),
                StructField("cardiacoutput", ArrayType(StringType()), False),
                StructField("cardiacinput", ArrayType(StringType()), False),
                StructField("svr", ArrayType(StringType()), False),
                StructField("svri", ArrayType(StringType()), False),
                StructField("pvr", ArrayType(StringType()), False),
                StructField("pvri", ArrayType(StringType()), False),
            ]
        )
        @F.pandas_udf(schema, functionType=F.PandasUDFType.GROUPED_MAP)
        def vitalAperiodic_unit(df):
            visit_id = str(df[self.visit_key].iloc[0])
            patient_id = self.visit_id_to_patient_id[visit_id]
            code = [None] * len(df)
            timestamp = [
                str(self.visit_id_to_encounter_time[visit_id] + pd.Timedelta(minutes=int(offset)))
                for offset in df["observationoffset"].tolist()
            ]
            noninvasivesystolic = df["noninvasivesystolic"].tolist()
            noninvasivediastolic = df["noninvasivediastolic"].tolist()
            noninvasivemean = df["noninvasivemean"].tolist()
            paop = df["paop"].tolist()
            cardiacoutput = df["cardiacoutput"].tolist()
            cardiacinput = df["cardiacinput"].tolist()
            svr = df["svr"].tolist()
            svri = df["svri"].tolist()
            pvr = df["pvr"].tolist()
            pvri = df["pvri"].tolist()
            return pd.DataFrame([[
                patient_id,
                visit_id,
                code,
                timestamp,
                noninvasivesystolic,
                noninvasivediastolic,
                noninvasivemean,
                paop,
                cardiacoutput,
                cardiacinput,
                svr,
                svri,
                pvr,
                pvri
            ]])
        
        pandas_df = spark_df.groupBy(self.visit_key).apply(vitalAperiodic_unit).toPandas()
        
        def aggregate_events(row):
            events = []
            p_id = row["patient_id"]
            v_id = row["visit_id"]
            for (
                code, timestamp, noninvasivesystolic, noninvasivediastolic, noninvasivemean,
                paop, cardiacoutput, cardiacinput, svr, svri, pvr, pvri
            ) in zip(
                row["code"], row["timestamp"], row["noninvasivesystolic"],
                row["noninvasivediastolic"], row["noninvasivemean"], row["paop"], row["cardiacoutput"],
                row["cardiacinput"], row["svr"], row["svri"], row["pvr"], row["pvri"]
            ):
                event = Event(
                    code=code,
                    table=table,
                    vocabulary="eicu_VITALAPERIODICSTRING",
                    visit_id=v_id,
                    patient_id=p_id,
                    timestamp=strptime(timestamp),
                    noninvasivesystolic=noninvasivesystolic,
                    noninvasivediastolic=noninvasivediastolic,
                    noninvasivemean=noninvasivemean,
                    paop=paop,
                    cardiacoutput=cardiacoutput,
                    cardiacinput=cardiacinput,
                    svr=svr,
                    svri=svri,
                    pvr=pvr,
                    pvri=pvri
                )
                events.append(event)
            return events

        # parallel apply to aggregate events
        aggregated_events = pandas_df.parallel_apply(aggregate_events, axis=1)
        
        # summarize the results
        patients = self._add_events_to_patient_dict(patients, aggregated_events)
        return patients

if __name__ == "__main__":
    dataset = eICUDataset(
        root="/srv/local/data/physionet.org/files/eicu-crd/2.0",
        tables=["diagnosis", "medication", "lab", "treatment", "physicalExam", "admissionDx"],
        refresh_cache=True,
    )
    dataset.stat()
    dataset.info()
