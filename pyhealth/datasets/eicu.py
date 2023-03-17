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
from pyhealth.datasets.utils import strptime

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
            for patients
        - treatment: contains treatment information (eICU_TREATMENTSTRING code)
            for patients.
        - medication: contains medication related order entries (eICU_DRUGNAME
            code) for patients.
        - lab: contains laboratory measurements (eICU_LABNAME code)
            for patients
        - physicalExam: contains all physical exam (eICU_PHYSICALEXAMPATH)
            conducted for patients.

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
        ...         tables=["diagnosis", "medication", "lab", "treatment", "physicalExam"],
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
            dtype={"patientunitstayid": str, "icd9code": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["patientunitstayid", "icd9code"])
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
            for offset, codes in zip(v_info["diagnosisoffset"], v_info["icd9code"]):
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
    """TODO: to be written"""

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

    def parse_diagnosis(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """TODO: to be written"""
        #TODO
        raise NotImplementedError()

    def parse_treatment(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """TODO: to be written"""
        #TODO
        raise NotImplementedError()

    def parse_medication(
        self,
        patients: Dict[str, Patient],
        spark: SparkSession
    ) -> Dict[str, Patient]:
        """TODO: to be written"""
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
        """TODO: to be written"""
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

    def parse_physicalexam(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """TODO: to be written"""
        #TODO
        raise NotImplementedError()

    def parse_infusiondrug(
        self,
        patients: Dict[str, Patient],
        spark: SparkSession
    ) -> Dict[str, Patient]:
        """TODO: to be written"""
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
        def infusiondrug_unit(df):
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

        pandas_df = spark_df.groupBy(self.visit_key).apply(infusiondrug_unit).toPandas()
        
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

if __name__ == "__main__":
    dataset = eICUDataset(
        root="/srv/local/data/physionet.org/files/eicu-crd/2.0",
        tables=["diagnosis", "medication", "lab", "treatment", "physicalExam"],
        refresh_cache=True,
    )
    dataset.stat()
    dataset.info()
