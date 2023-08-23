import warnings
import os
from typing import Optional, List, Dict, Union, Tuple

import pandas as pd
import datetime

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, ArrayType, StringType

from pyhealth.data import Event, Visit, Patient
from pyhealth.datasets import BaseEHRDataset, BaseEHRSparkDataset
from pyhealth.datasets.utils import strptime

# TODO: add other tables


class MIMIC4Dataset(BaseEHRDataset):
    """Base dataset for MIMIC-IV dataset.

    The MIMIC-IV dataset is a large dataset of de-identified health records of ICU
    patients. The dataset is available at https://mimic.physionet.org/.

    The basic information is stored in the following tables:
        - patients: defines a patient in the database, subject_id.
        - admission: define a patient's hospital admission, hadm_id.

    We further support the following tables:
        - diagnoses_icd: contains ICD diagnoses (ICD9CM and ICD10CM code)
            for patients.
        - procedures_icd: contains ICD procedures (ICD9PROC and ICD10PROC
            code) for patients.
        - prescriptions: contains medication related order entries (NDC code)
            for patients.
        - labevents: contains laboratory measurements (MIMIC4_ITEMID code)
            for patients

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
        >>> from pyhealth.datasets import MIMIC4Dataset
        >>> dataset = MIMIC4Dataset(
        ...         root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",
        ...         tables=["diagnoses_icd", "procedures_icd", "prescriptions", "labevents"],
        ...         code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
        ...     )
        >>> dataset.stat()
        >>> dataset.info()
    """

    def parse_basic_info(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper functions which parses patients and admissions tables.

        Will be called in `self.parse_tables()`

        Docs:
            - patients:https://mimic.mit.edu/docs/iv/modules/hosp/patients/
            - admissions: https://mimic.mit.edu/docs/iv/modules/hosp/admissions/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        # read patients table
        patients_df = pd.read_csv(
            os.path.join(self.root, "patients.csv"),
            dtype={"subject_id": str},
            nrows=1000 if self.dev else None,
        )
        # read admissions table
        admissions_df = pd.read_csv(
            os.path.join(self.root, "admissions.csv"),
            dtype={"subject_id": str, "hadm_id": str},
        )
        # merge patients and admissions tables
        df = pd.merge(patients_df, admissions_df, on="subject_id", how="inner")
        # sort by admission and discharge time
        df = df.sort_values(["subject_id", "admittime", "dischtime"], ascending=True)
        # group by patient
        df_group = df.groupby("subject_id")

        # parallel unit of basic information (per patient)
        def basic_unit(p_id, p_info):
            # no exact birth datetime in MIMIC-IV
            # use anchor_year and anchor_age to approximate birth datetime
            anchor_year = int(p_info["anchor_year"].values[0])
            anchor_age = int(p_info["anchor_age"].values[0])
            birth_year = anchor_year - anchor_age
            patient = Patient(
                patient_id=p_id,
                # no exact month, day, and time, use Jan 1st, 00:00:00
                birth_datetime=strptime(str(birth_year)),
                # no exact time, use 00:00:00
                death_datetime=strptime(p_info["dod"].values[0]),
                gender=p_info["gender"].values[0],
                ethnicity=p_info["race"].values[0],
                anchor_year_group=p_info["anchor_year_group"].values[0],
            )
            # load visits
            for v_id, v_info in p_info.groupby("hadm_id"):
                visit = Visit(
                    visit_id=v_id,
                    patient_id=p_id,
                    encounter_time=strptime(v_info["admittime"].values[0]),
                    discharge_time=strptime(v_info["dischtime"].values[0]),
                    discharge_status=v_info["hospital_expire_flag"].values[0],
                )
                # add visit
                patient.add_visit(visit)
            return patient

        # parallel apply
        df_group = df_group.parallel_apply(
            lambda x: basic_unit(x.subject_id.unique()[0], x)
        )
        # summarize the results
        for pat_id, pat in df_group.items():
            patients[pat_id] = pat

        return patients

    def parse_diagnoses_icd(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses diagnosis_icd table.

        Will be called in `self.parse_tables()`

        Docs:
            - diagnosis_icd: https://mimic.mit.edu/docs/iv/modules/hosp/diagnoses_icd/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.

        Note:
            MIMIC-IV does not provide specific timestamps in diagnoses_icd
                table, so we set it to None.
        """
        table = "diagnoses_icd"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"subject_id": str, "hadm_id": str, "icd_code": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["subject_id", "hadm_id", "icd_code", "icd_version"])
        # sort by sequence number (i.e., priority)
        df = df.sort_values(["subject_id", "hadm_id", "seq_num"], ascending=True)
        # group by patient and visit
        group_df = df.groupby("subject_id")

        # parallel unit of diagnosis (per patient)
        def diagnosis_unit(p_id, p_info):
            events = []
            # iterate over each patient and visit
            for v_id, v_info in p_info.groupby("hadm_id"):
                for code, version in zip(v_info["icd_code"], v_info["icd_version"]):
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary=f"ICD{version}CM",
                        visit_id=v_id,
                        patient_id=p_id,
                    )
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: diagnosis_unit(x.subject_id.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def parse_procedures_icd(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses procedures_icd table.

        Will be called in `self.parse_tables()`

        Docs:
            - procedures_icd: https://mimic.mit.edu/docs/iv/modules/hosp/procedures_icd/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.

        Note:
            MIMIC-IV does not provide specific timestamps in procedures_icd
                table, so we set it to None.
        """
        table = "procedures_icd"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"subject_id": str, "hadm_id": str, "icd_code": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["subject_id", "hadm_id", "icd_code", "icd_version"])
        # sort by sequence number (i.e., priority)
        df = df.sort_values(["subject_id", "hadm_id", "seq_num"], ascending=True)
        # group by patient and visit
        group_df = df.groupby("subject_id")

        # parallel unit of procedure (per patient)
        def procedure_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("hadm_id"):
                for code, version in zip(v_info["icd_code"], v_info["icd_version"]):
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary=f"ICD{version}PROC",
                        visit_id=v_id,
                        patient_id=p_id,
                    )
                    # update patients
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: procedure_unit(x.subject_id.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)

        return patients

    def parse_prescriptions(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses prescriptions table.

        Will be called in `self.parse_tables()`

        Docs:
            - prescriptions: https://mimic.mit.edu/docs/iv/modules/hosp/prescriptions/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "prescriptions"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            low_memory=False,
            dtype={"subject_id": str, "hadm_id": str, "ndc": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["subject_id", "hadm_id", "ndc"])
        # sort by start date and end date
        df = df.sort_values(
            ["subject_id", "hadm_id", "starttime", "stoptime"], ascending=True
        )
        # group by patient and visit
        group_df = df.groupby("subject_id")

        # parallel unit of prescription (per patient)
        def prescription_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("hadm_id"):
                for timestamp, code in zip(v_info["starttime"], v_info["ndc"]):
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary="NDC",
                        visit_id=v_id,
                        patient_id=p_id,
                        timestamp=strptime(timestamp),
                    )
                    # update patients
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: prescription_unit(x.subject_id.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)

        return patients

    def parse_labevents(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses labevents table.

        Will be called in `self.parse_tables()`

        Docs:
            - labevents: https://mimic.mit.edu/docs/iv/modules/hosp/labevents/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "labevents"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"subject_id": str, "hadm_id": str, "itemid": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["subject_id", "hadm_id", "itemid"])
        # sort by charttime
        df = df.sort_values(["subject_id", "hadm_id", "charttime"], ascending=True)
        # group by patient and visit
        group_df = df.groupby("subject_id")

        # parallel unit of labevent (per patient)
        def lab_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("hadm_id"):
                for timestamp, code in zip(v_info["charttime"], v_info["itemid"]):
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary="MIMIC4_ITEMID",
                        visit_id=v_id,
                        patient_id=p_id,
                        timestamp=strptime(timestamp),
                    )
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: lab_unit(x.subject_id.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def parse_hcpcsevents(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses hcpcsevents table.

        Will be called in `self.parse_tables()`

        Docs:
            - hcpcsevents: https://mimic.mit.edu/docs/iv/modules/hosp/hcpcsevents/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.

        Note:
            MIMIC-IV does not provide specific timestamps in hcpcsevents
                table, so we set it to None.
        """
        table = "hcpcsevents"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"subject_id": str, "hadm_id": str, "hcpcs_cd": str},
        )
        # drop rows with missing values
        df = df.dropna(subset=["subject_id", "hadm_id", "hcpcs_cd"])
        # sort by sequence number (i.e., priority)
        df = df.sort_values(["subject_id", "hadm_id", "seq_num"], ascending=True)
        # group by patient and visit
        group_df = df.groupby("subject_id")

        # parallel unit of hcpcsevents (per patient)
        def hcpcsevents_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("hadm_id"):
                for code in v_info["hcpcs_cd"]:
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary="MIMIC4_HCPCS_CD",
                        visit_id=v_id,
                        patient_id=p_id,
                    )
                    # update patients
                    events.append(event)
            return events
            
        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: hcpcsevents_unit(x.subject_id.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        
        return patients

class MIMIC4SparkDataset(BaseEHRSparkDataset):
    """Base dataset for MIMIC-IV dataset utilized by PySpark for the efficient data processing.

    The MIMIC-IV dataset is a large dataset of de-identified health records of ICU
    patients. The dataset is available at https://mimic.physionet.org/.

    Unlike the normal mimic4 dataset (MIMIC4Dataset) that does not utilize PySpark, this
    dataset provides not only the corresponding code for each event but also any other
    available columns by attr_dict or internal attributes to support further data processing.

    The basic information is stored in the following tables:
        - patients: defines a patient in the database, subject_id.
        - admission: defines a patient's hospital admission, hadm_id.
        - icustays: defines a patient's ICU stay, stay_id.
    
    We further support the following tables:
        - diagnoses_icd: contains ICD diagnoses (ICD9CM and ICD10CM code) for patients.
        - procedures_icd: contains ICD procedures (ICD9PROC and ICd10PROC code) for patients.
        - prescriptions: contains medication related order entries (NCD code) for patients.
        - labevents: contains laboratory measurements (MIMIC4_ITEMID code) for patients.
        - chartevents: contains all charted data (MIMIC4_ITEMID code) for patients.
        - microbiologyevents: contains microbiology information for patients, including cultures 
            acquired and associated sensitivities. Note that we set `code` to `None` for all events
            from this table because there is no specific column for `code` in microbiologyevents
            table.
        - inputevents: contains input data (MIMIC4_ITEMID code) for patients.
        - outputevents: contains output data (MIMIC4_ITEMID code) for patients.
        - procedureevents: contains procedures (MIMIC4_ITEMID code) for patients.
    
    Args:
        dataset_name: name of the dataset.
        root: root directory of the raw data (should contain many csv files).
        tables: list of tables to be loaded (e.g., ["LABEVENTS", "PRESCRIPTIONS"]). Basic
            tables will be loaded by default.
        visit_unit: unit of visit to be grouped. Available options are typed in VISIT_UNIT_CHOICES.
            Default is "hospital", which means to regard each hospital admission as a visit.
        observation_size: size of the observation window. only the events within the first N hours
            are included in the processed data. Default is 12.
        gap_size: size of gap window. labels of some prediction tasks (E.g., short-term mortality
            prediction) are defined between `observation_size` + `gap_size` and the next N hours of
            `prediction_size`. If a patient's stay has less than `observaion_size` + `gap_size`
            duration, some tasks cannot be defined for that stay. Default is 0.
        prediction_size: size of prediction window. labels of some prediction tasks (E.g., short-term mortality
            prediction) are defined between `observation_size` + `gap_size` and the next N hours of
            `prediction_size`. Default is 24.
        discard_samples_with_missing_label: whether to discard samples with any missing label (-1)
            when defining tasks. Default is False, which assigns -1 to the sample on that task.
        code_mapping: a dictionary containing the code mapping information.
            The key is a str of the source code vocabulary and the value is of
            two formats:
                (1) a str of the target code vocabulary. E.g., {"NDC", "ATC"}.
                (2) a tuple with two elements. The first element is a str of the
                    target code vocabulary and the second element is a dict with
                    keys "source_kwargs" or "target_kwargs" and values of the
                    corresponding kwargs for the `CrossMap.map()` method.
                Default is empty dict, which means the original code will be used.
        dev: whether to enable dev mode (only use a small subset of the data).
            Default is False.
        refresh_cache: whether to refresh the cache; if true, the dataset will
            be processed from scratch and the cache will be updated. Default is False.

    Examples:
        >>> from pyhealth.datasets import MIMIC4SparkDataset
        >>> dataset = MIMIC4SparkDataset(
        ...     root="/usr/local/data/physionet.org/files/mimiciv/2.0",
        ...     tables=["LABEVENTS", "PRESCRIPTIONS"],
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
        visit_unit,
        **kwargs
    ):
        if visit_unit == "hospital":
            self.visit_key = "hadm_id"
            self.encounter_key = "admittime"
            self.discharge_key = "dischtime"
        elif visit_unit == "icu":
            self.visit_key = "stay_id"
            self.encounter_key = "intime"
            self.discharge_key = "outtime"
        super().__init__(root, tables, visit_unit, **kwargs)

    def parse_basic_info(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper functions which parses patients and admissions tables.

        Will be called in `self.parse_tables()`

        Docs:
            - patients: https://mimic.mit.edu/docs/iv/modules/hosp/patients/
            - admissions: https://mimic.mit.edu/docs/iv/modules/hosp/admissions/
            - icustays: https://mimic.mit.edu/docs/iv/modules/icu/icustays/
            - diagnoses_icd: https://mimic.mit.edu/docs/iv/modules/hosp/diagnoses_icd/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        # read patients table
        if os.path.exists(os.path.join(self.root, "patients.csv")):
            path = os.path.join(self.root, "patients.csv")
        else:
            path = os.path.join(self.root, "hosp/patients.csv")
        patients_df = pd.read_csv(path, dtype={"subject_id": str}, nrows=1000 if self.dev else None)
        # read admissions table
        if os.path.exists(os.path.join(self.root, "admissions.csv")):
            path = os.path.join(self.root, "admissions.csv")
        else:
            path = os.path.join(self.root, "hosp/admissions.csv")
        admissions_df = pd.read_csv(path, dtype={"subject_id": str, "hadm_id": str})
        # read icustays table
        if os.path.exists(os.path.join(self.root, "icustays.csv")):
            path = os.path.join(self.root, "icustays.csv")
        else:
            path = os.path.join(self.root, "icu/icustays.csv")
        icustays_df = pd.read_csv(path, dtype={"hadm_id": str, "stay_id": str})
        # read diagnoses_icd table
        if os.path.exists(os.path.join(self.root, "diagnoses_icd.csv")):
            path = os.path.join(self.root, "diagnoses_icd.csv")
        else:
            path = os.path.join(self.root, "hosp/diagnoses_icd.csv")
        diagnoses_df = pd.read_csv(path, dtype={"hadm_id": str})
        diagnoses_df = diagnoses_df.groupby(
            "hadm_id"
        )[["icd_code", "icd_version"]].agg(list)

        # merge patient, admission, and icustay tables
        df = pd.merge(
            patients_df,
            admissions_df,
            on="subject_id",
            how="inner"
        )
        df = pd.merge(
            df,
            icustays_df[["hadm_id", "stay_id", "intime", "outtime"]],
            on="hadm_id",
            how="inner"
        )
        # merge with diagnoses_icd table
        df = pd.merge(
            df,
            diagnoses_df,
            on="hadm_id",
            how="inner"
        )
        # sort by admission and discharge time
        df = df.sort_values(["subject_id", self.encounter_key, self.discharge_key], ascending=True)
        # group by patient
        df_group = df.groupby("subject_id")

        # parallel unit of basic information (per patient)
        def basic_unit(p_id, p_info):
            # no exact birth datetime in MIMIC-IV
            # use anchor_year and anchor_age to approximate birth datetime
            anchor_year = int(p_info["anchor_year"].values[0])
            anchor_age = int(p_info["anchor_age"].values[0])
            birth_year = anchor_year - anchor_age
            patient = Patient(
                patient_id=p_id,
                # no exact month, day, and time, use Jan 1st, 00:00:00
                birth_datetime=strptime(str(birth_year)),
                # no exact time, use 00:00:00
                death_datetime=strptime(p_info["dod"].values[0]),
                gender=p_info["gender"].values[0],
                ethnicity=p_info["race"].values[0],
                anchor_year_group=p_info["anchor_year_group"].values[0],
            )
            # load visits
            for v_id, v_info in p_info.groupby(self.visit_key):
                visit = Visit(
                    visit_id=v_id,
                    patient_id=p_id,
                    encounter_time=strptime(v_info[self.encounter_key].values[0]),
                    discharge_time=strptime(v_info[self.discharge_key].values[0]),
                    discharge_status=v_info["hospital_expire_flag"].values[0],
                    discharge_location=v_info["discharge_location"].values[0],
                    hadm_id=v_info["hadm_id"].values[0],
                    hospital_discharge_time=strptime(v_info["dischtime"].values[0]),
                    diagnosis_codes=v_info["icd_code"].values[0],
                    icd_versions = v_info["icd_version"].values[0]
                )
                # add visit
                patient.add_visit(visit)
            return patient

        # parallel apply
        df_group = df_group.parallel_apply(
            lambda x: basic_unit(x.subject_id.unique()[0], x)
        )

        # summarize the results
        for pat_id, pat in df_group.items():
            patients[pat_id] = pat

        return patients

    def parse_diagnoses_icd(
        self,
        patients: Dict[str, Patient],
        spark: SparkSession,
    ) -> Dict[str, Patient]:
        """Helper function which parses diagnoses_icd table.
        
        Will be called in `self.parse_tables()`.
        
        Docs:
            - diagnoses_icd: https://mimic.mit.edu/docs/iv/modules/hosp/diagnoses_icd/
        
        Args:
            patients: a dict of `Patient` objects indexed by patient_id.
            spark: a spark session for reading the table.
        
        Returns:
            The updated patients dict.
        
        Note:
            MIMIC-IV does not provide specific timestamps in diagnoses_icd table,
                so we set it to "the end of the hospital stay".
        """
        table = "diagnoses_icd"
        
        spark_df = spark.read.csv(os.path.join(self.root, f"{table}.csv"), header=True)
        if self.dev:
            spark_df = spark_df.limit(3000)
        
        if self.visit_unit == "icu":
            raise ValueError(
                "Cannot determine icu-level diagnosis events since the ICD codes are generated at "
                "the end of the hospital stay. Please exclude diagnoses_icd table or instantiate "
                ":MIMIC4SparkDataset: class with hospital-level by specifying "
                "`visit_unit='hospital'`."
            )
        else:
            spark_df = self.filter_events(
                spark=spark,
                spark_df=spark_df,
                timestamp_key=None,
                table=table,
                joined_table="admissions",
                select=["hadm_id", "admittime", "dischtime"],
                on="hadm_id",
                should_infer_icustay_id=False,
            )
        
        schema = StructType(
            [
                StructField("patient_id", StringType(), False),
                StructField("visit_id", StringType(), False),
                StructField("code", ArrayType(StringType()), False),
                StructField("name", ArrayType(StringType()), False),
                StructField("timestamp", ArrayType(StringType()), False),
                StructField("seq_num", ArrayType(StringType()), False),
                StructField("icd_version", ArrayType(StringType()), False)
            ]
        )
        @F.pandas_udf(schema, functionType=F.PandasUDFType.GROUPED_MAP)
        def diagnoses_icd_unit(df):
            patient_id = str(df["subject_id"].iloc[0])
            visit_id = str(df[self.visit_key].iloc[0])
            code = df["icd_code"].tolist()
            name = ["icd_code"] * len(code)
            # NOTE: diagnosis codes are generated at the end of the hospital stay.
            timestamp = [str(t) for t in df["dischtime"].tolist()]
            seq_num = df["seq_num"].tolist()
            icd_version = df["icd_version"].tolist()
            return pd.DataFrame([[
                patient_id,
                visit_id,
                code,
                name,
                timestamp,
                seq_num,
                icd_version
            ]])
        
        pandas_df = spark_df.groupBy(
            "subject_id", self.visit_key
        ).apply(diagnoses_icd_unit).toPandas()
        
        def aggregate_events(row):
            events = []
            p_id = row["patient_id"]
            v_id = row["visit_id"]
            for code, name, timestamp, seq_num, icd_version in zip(
                row["code"], row["name"], row["timestamp"], row["seq_num"], row["icd_version"]
            ):
                event = Event(
                    code=code,
                    name=name,
                    table=table,
                    vocabulary=f"ICD{icd_version}CM",
                    visit_id=v_id,
                    patient_id=p_id,
                    timestamp=strptime(timestamp),
                    seq_num=seq_num,
                )
                events.append(event)
            return events

        # parallel apply to aggergate events
        aggregated_events = pandas_df.parallel_apply(aggregate_events, axis=1)
        
        # summarize the results
        patients = self._add_events_to_patient_dict(patients, aggregated_events)
        return patients

    def parse_procedures_icd(
        self,
        patients: Dict[str, Patient],
        spark: SparkSession
    ) -> Dict[str, Patient]:
        """Helper function which parses procedures_icd table.
        
        Will be called in `self.parse_tables()`.
        
        Docs:
            - procedures_icd: https://mimic.mit.edu/docs/iv/modules/hosp/procedures_icd/
        
        Args:
            patients: a dict of `Patient` objects indexed by patient_id.
            spark: a spark session for reading the table.
        
        Returns:
            The updated patients dict.
        
        Note:
            MIMIC-IV provides only the date of the associated procedures,
                so we set it to "the midnight (00:00:00) of the chartdate".
        """
        table = "procedures_icd"
        
        # read table
        spark_df = spark.read.csv(os.path.join(self.root, f"{table}.csv"), header=True)
        if self.dev:
            spark_df = spark_df.limit(3000)
        # convert dtype of charttime to datetime
        spark_df = spark_df.withColumn("chartdate", F.to_timestamp("chartdate"))
        
        if self.visit_unit == "icu":
            warnings.warn(
                "We treat the timestamps for procedure_icd events as the midnight (00:00:00) of "
                "their chartdate since procedures_icd table provides only the date of the "
                "associated procedures."
            )
            spark_df = self.filter_events(
                spark=spark,
                spark_df=spark_df,
                timestamp_key="chartdate",
                joined_table="icustays",
                select=["hadm_id", "stay_id", "intime", "outtime"],
                on="hadm_id",
                should_infer_icustay_id=True,
            )
        else:
            spark_df = self.filter_events(
                spark=spark,
                spark_df=spark_df,
                timestamp_key=None,
                table=table,
                joined_table="admissions",
                select=["hadm_id", "admittime", "dischtime"],
                on="hadm_id",
                should_infer_icustay_id=False,
            )

        # sort by chartdate
        spark_df = spark_df.sort(["subject_id", self.visit_key, "chartdate"], ascending=True)

        schema = StructType(
            [
                StructField("patient_id", StringType(), False),
                StructField("visit_id", StringType(), False),
                StructField("code", ArrayType(StringType()), False),
                StructField("name", ArrayType(StringType()), False),
                StructField("timestamp", ArrayType(StringType()), False),
                StructField("seq_num", ArrayType(StringType()), False),
                StructField("icd_version", ArrayType(StringType()), False),
            ]
        )
        @F.pandas_udf(schema, functionType=F.PandasUDFType.GROUPED_MAP)
        def procedures_icd_unit(df):
            patient_id = str(df["subject_id"].iloc[0])
            visit_id = str(df[self.visit_key].iloc[0])
            code = df["icd_code"]
            name = ["icd_code"] * len(code)
            timestamp = [str(t) for t in df["chartdate"].tolist()]
            seq_num = df["seq_num"].tolist()
            icd_version = df["icd_version"].tolist()
            return pd.DataFrame([[
                patient_id,
                visit_id,
                code,
                name,
                timestamp,
                seq_num,
                icd_version
            ]])
        
        pandas_df = spark_df.groupBy(
            "subject_id", self.visit_key
        ).apply(procedures_icd_unit).toPandas()
        
        def aggregate_events(row):
            events = []
            p_id = row["patient_id"]
            v_id = row["visit_id"]
            for code, name, timestamp, seq_num, icd_version in zip(
                row["code"], row["name"], row["timestamp"], row["seq_num"], row["icd_version"]
            ):
                event = Event(
                    code=code,
                    name=name,
                    table=table,
                    vocabulary=f"ICD{icd_version}PROC",
                    visit_id=v_id,
                    patient_id=p_id,
                    timestamp=strptime(timestamp),
                    seq_num=seq_num
                )
                events.append(event)
            return events
        
        # parallel apply to aggregate events
        aggregated_events = pandas_df.parallel_apply(aggregate_events, axis=1)
        
        # summarize the results
        patients = self._add_events_to_patient_dict(patients, aggregated_events)
        return patients

    def parse_prescriptions(
        self,
        patients: Dict[str, Patient],
        spark: SparkSession
    ) -> Dict[str, Patient]:
        """Helper function which parses prescriptions table.
        
        Will be called in `self.parse_tables()`.
        
        Docs:
            - prescriptions: https://mimic.mit.edu/docs/iv/modules/hosp/prescriptions/
        
        Args:
            patients: a dict of `Patient` objects indexed by patient_id.
            spark: a spark session for reading the table.
        
        Returns:
            The updated patients dict.
        
        """
        table = "prescriptions"
        
        spark_df = spark.read.csv(os.path.join(self.root, f"{table}.csv"), header=True)
        if self.dev:
            spark_df = spark_df.limit(3000)
        # convert dtype of starttime to datetime
        spark_df = spark_df.withColumn("starttime", F.to_timestamp("starttime"))
        
        if self.visit_unit == "icu":
            spark_df = self.filter_events(
                spark=spark,
                spark_df=spark_df,
                timestamp_key="starttime",
                joined_table="icustays",
                select=["hadm_id", "stay_id", "intime", "outtime"],
                on="hadm_id",
                should_infer_icustay_id=True,
            )
        else:
            spark_df = self.filter_events(
                spark=spark,
                spark_df=spark_df,
                timestamp_key="starttime",
                joined_table="admissions",
                select=["hadm_id", "admittime", "dischtime"],
                on="hadm_id",
                should_infer_icustay_id=False,
            )

        # sort by starttime
        spark_df = spark_df.sort(
            ["subject_id", self.visit_key, "starttime", "stoptime"],
            ascending=True
        )
        
        schema = StructType(
            [
                StructField("patient_id", StringType(), False),
                StructField("visit_id", StringType(), False),
                StructField("code", ArrayType(StringType()), False),
                StructField("name", ArrayType(StringType()), False),
                StructField("timestamp", ArrayType(StringType()), False),
                StructField("pharmacy_id", ArrayType(StringType()), False),
                StructField("poe_id", ArrayType(StringType()), False),
                StructField("poe_seq", ArrayType(StringType()), False),
                StructField("drug_type", ArrayType(StringType()), False),
                StructField("drug", ArrayType(StringType()), False),
                StructField("formulary_drug_cd", ArrayType(StringType()), False),
                StructField("gsn", ArrayType(StringType()), False),
                StructField("prod_strength", ArrayType(StringType()), False),
                StructField("form_rx", ArrayType(StringType()), False),
                StructField("dose_val_rx", ArrayType(StringType()), False),
                StructField("dose_unit_rx", ArrayType(StringType()), False),
                StructField("form_val_disp", ArrayType(StringType()), False),
                StructField("form_unit_disp", ArrayType(StringType()), False),
                StructField("doses_per_24_hrs", ArrayType(StringType()), False),
                StructField("route", ArrayType(StringType()), False)
            ]
        )
        @F.pandas_udf(schema, functionType=F.PandasUDFType.GROUPED_MAP)
        def prescription_unit(df):
            patient_id = str(df["subject_id"].iloc[0])
            visit_id = str(df[self.visit_key].iloc[0])
            code = df["ndc"].tolist()
            name = ["ndc"] * len(code)
            timestamp = [str(t) for t in df["starttime"].tolist()]
            pharmacy_id = df["pharmacy_id"].tolist()
            poe_id = df["poe_id"].tolist()
            poe_seq = df["poe_seq"].tolist()
            drug_type = df["drug_type"].tolist()
            drug = df["drug"].tolist()
            formulary_drug_cd = df["formulary_drug_cd"].tolist()
            gsn = df["gsn"].tolist()
            prod_strength = df["prod_strength"].tolist()
            form_rx = df["form_rx"].tolist()
            dose_val_rx = df["dose_val_rx"].tolist()
            dose_unit_rx = df["dose_unit_rx"].tolist()
            form_val_disp = df["form_val_disp"].tolist()
            form_unit_disp = df["form_unit_disp"].tolist()
            doses_per_24_hrs = df["doses_per_24_hrs"].tolist()
            route = df["route"].tolist()
            return pd.DataFrame([[
                patient_id,
                visit_id,
                code,
                timestamp,
                pharmacy_id,
                poe_id,
                poe_seq,
                drug_type,
                drug,
                formulary_drug_cd,
                gsn,
                prod_strength,
                form_rx,
                dose_val_rx,
                dose_unit_rx,
                form_val_disp,
                form_unit_disp,
                doses_per_24_hrs,
                route
            ]])

        pandas_df = spark_df.groupBy(
            "subject_id", self.visit_key
        ).apply(prescription_unit).toPandas()
        
        def aggregate_events(row):
            events = []
            p_id = row["patient_id"]
            v_id = row["visit_id"]
            for (
                code, name, timestamp, pharmacy_id, poe_id, poe_seq, drug_type, drug,
                formulary_drug_cd, gsn, prod_strength, form_rx, dose_val_rx, dose_unit_rx,
                form_val_disp, form_unit_disp, doses_per_24_hrs, route
            ) in zip(
                row["code"], row["name"], row["timestamp"], row["pharmacy_id"], row["poe_id"],
                row["poe_seq"], row["drug_type"], row["drug"], row["formulary_drug_cd"], row["gsn"],
                row["prod_strength"], row["form_rx"], row["dose_val_rx"], row["dose_unit_rx"],
                row["form_val_disp"], row["form_unit_disp"], row["doses_per_24_hrs"], row["route"]
            ):
                event = Event(
                    code=code,
                    name=name,
                    table=table,
                    vocabulary="NDC",
                    visit_id=v_id,
                    patient_id=p_id,
                    timestamp=strptime(timestamp),
                    pharmacy_id=pharmacy_id,
                    poe_id=poe_id,
                    poe_seq=poe_seq,
                    drug_type=drug_type,
                    drug=drug,
                    formulary_drug_cd=formulary_drug_cd,
                    gsn=gsn,
                    prod_strength=prod_strength,
                    form_rx=form_rx,
                    dose_val_rx=dose_val_rx,
                    dose_unit_rx=dose_unit_rx,
                    form_val_disp=form_val_disp,
                    form_unit_disp=form_unit_disp,
                    doses_per_24_hrs=doses_per_24_hrs,
                    route=route
                )
                events.append(event)
            return events
        
        # parallel apply to aggregate events
        aggregated_events = pandas_df.parallel_apply(aggregate_events, axis=1)
        
        # summarize the results
        patients = self._add_events_to_patient_dict(patients, aggregated_events)
        return patients

    def parse_labevents(
        self,
        patients: Dict[str, Patient],
        spark: SparkSession
    ) -> Dict[str, Patient]:
        """Helper functions which parses labevents table.
        
        Will be called in `self.parse_tables()`.
        
        Docs:
            - labevents: https://mimic.mit.edu/docs/iv/modules/hosp/labevents/
        
        Args:
            patients: a dict of `Patient` objetcs indexed by patient_id.
            spark: a spark session for reading the table.
        
        Returns:
            The updated patients dict.
        """
        table = "labevents"
        
        # read table
        spark_df = spark.read.csv(os.path.join(self.root, f"{table}.csv"), header=True)
        if self.dev:
            spark_df = spark_df.limit(3000)
        # convert dtype of charttime to datetime
        spark_df = spark_df.withColumn("charttime", F.to_timestamp("charttime"))
        
        if self.visit_unit == "icu":
            spark_df = self.filter_events(
                spark=spark,
                spark_df=spark_df,
                timestamp_key="charttime",
                joined_table="icustays",
                select=["hadm_id", "stay_id", "intime", "outtime"],
                on="hadm_id",
                should_infer_icustay_id=True,
            )
        else:
            spark_df = self.filter_events(
                spark=spark,
                spark_df=spark_df,
                timestamp_key="charttime",
                joined_table="admissions",
                select=["hadm_id", "admittime", "dischtime"],
                on="hadm_id",
                should_infer_icustay_id=False,
            )
        
        # sort by charttime
        spark_df = spark_df.sort(["subject_id", self.visit_key, "charttime"], ascending=True)

        schema = StructType(
            [
                StructField("patient_id", StringType(), False),
                StructField("visit_id", StringType(), False),
                StructField("code", ArrayType(StringType()), False),
                StructField("name", ArrayType(StringType()), False),
                StructField("timestamp", ArrayType(StringType()), False),
                StructField("value", ArrayType(StringType()), False),
                StructField("valuenum", ArrayType(StringType()), False),
                StructField("valueuom", ArrayType(StringType()), False),
                StructField("ref_range_lower", ArrayType(StringType()), False),
                StructField("ref_range_upper", ArrayType(StringType()), False),
                StructField("flag", ArrayType(StringType()), False),
                StructField("priority", ArrayType(StringType()), False),
                StructField("comments", ArrayType(StringType()), False)
            ]
        )
        @F.pandas_udf(schema, functionType=F.PandasUDFType.GROUPED_MAP)
        def lab_unit(df):
            patient_id = str(df["subject_id"].iloc[0])
            visit_id = str(df[self.visit_key].iloc[0])
            code = df["itemid"].tolist()
            name = ["itemid"] * len(code)
            timestamp = [str(t) for t in df["charttime"].tolist()]
            value = df["value"].tolist()
            valuenum = df["valuenum"].tolist()
            valueuom = df["valueuom"].tolist()
            ref_range_lower = df["ref_range_lower"].tolist()
            ref_range_upper = df["ref_range_upper"].tolist()
            flag = df["flag"].tolist()
            priority = df["priority"].tolist()
            comments = df["comments"].tolist()
            return pd.DataFrame([[
                patient_id,
                visit_id,
                code,
                name,
                timestamp,
                value,
                valuenum,
                valueuom,
                ref_range_lower,
                ref_range_upper,
                flag,
                priority,
                comments
            ]])

        pandas_df = spark_df.groupBy("subject_id", self.visit_key).apply(lab_unit).toPandas()
        
        def aggregate_events(row):
            events = []
            p_id = row["patient_id"]
            v_id = row["visit_id"]
            for (
                code, name, timestamp, value, valuenum, valueuom, ref_range_lower, ref_range_upper,
                flag, priority, comments
            ) in zip(
                row["code"], row["name"], row["timestamp"], row["value"], row["valuenum"],
                row["valueuom"], row["ref_range_lower"], row["ref_range_upper"], row["flag"],
                row["priority"], row["comments"]
            ):
                event = Event(
                    code=code,
                    name=name,
                    table=table,
                    vocabulary="MIMIC4_ITEMID",
                    visit_id=v_id,
                    patient_id=p_id,
                    timestamp=strptime(timestamp),
                    value=value,
                    valuenum=valuenum,
                    valueuom=valueuom,
                    ref_range_lower=ref_range_lower,
                    ref_range_upper=ref_range_upper,
                    flag=flag,
                    priority=priority,
                    comments=comments
                )
                events.append(event)
            return events

        # parallel apply to aggregate events
        aggregated_events = pandas_df.parallel_apply(aggregate_events, axis=1)
        
        # summarize the results
        patients = self._add_events_to_patient_dict(patients, aggregated_events)
        return patients

    def parse_inputevents(
        self,
        patients: Dict[str, Patient],
        spark: SparkSession
    ) -> Dict[str, Patient]:
        """Helper function which parses inputevents table.
        
        Will be called in `self.parse_tables()`.
        
        Docs:
            - inputevents: https://mimic.mit.edu/docs/iv/modules/icu/inputevents/
        
        Args:
            patients: a dict of `Patient` objects indexed by patient_id.
            spark: a spark session for reading the table.
        
        Returns:
            The updated patients dict.
        """
        table = "inputevents"
        
        # read table
        spark_df = spark.read.csv(os.path.join(self.root, f"{table}.csv"), header=True)
        if self.dev:
            spark_df = spark_df.limit(3000)
        # convert dtype of starttime
        spark_df = spark_df.withColumn("starttime", F.to_timestamp("starttime"))

        if self.visit_unit == "icu":
            spark_df = self.filter_events(
                spark=spark,
                spark_df=spark_df,
                timestamp_key="starttime",
                joined_table="icustays",
                select=["stay_id", "intime", "outtime"],
                on="stay_id",
                should_infer_icustay_id=False,
            )
        else:
            spark_df = self.filter_events(
                spark=spark,
                spark_df=spark_df,
                timestamp_key="starttime",
                joined_table="admissions",
                select=["hadm_id", "admittime", "dischtime"],
                on="hadm_id",
                should_infer_icustay_id=False,
            )
        
        # sort by starttime
        spark_df = spark_df.sort(
            ["subject_id", self.visit_key, "starttime", "endtime"],
            ascending=True
        )
        
        schema = StructType(
            [
                StructField("patient_id", StringType(), False),
                StructField("visit_id", StringType(), False),
                StructField("code", ArrayType(StringType()), False),
                StructField("name", ArrayType(StringType()), False),
                StructField("timestamp", ArrayType(StringType()), False),
                StructField("amount", ArrayType(StringType()), False),
                StructField("amountuom", ArrayType(StringType()), False),
                StructField("rate", ArrayType(StringType()), False),
                StructField("rateuom", ArrayType(StringType()), False),
                StructField("orderid", ArrayType(StringType()), False),
                StructField("linkorderid", ArrayType(StringType()), False),
                StructField("ordercategoryname", ArrayType(StringType()), False),
                StructField("secondaryordercategoryname", ArrayType(StringType()), False),
                StructField("ordercomponenttypedescription", ArrayType(StringType()), False),
                StructField("ordercategorydescription", ArrayType(StringType()), False),
                StructField("patientweight", ArrayType(StringType()), False),
                StructField("totalamount", ArrayType(StringType()), False),
                StructField("totalamountuom", ArrayType(StringType()), False),
                StructField("isopenbag", ArrayType(StringType()), False),
                StructField("continueinnextdept", ArrayType(StringType()), False),
                StructField("statusdescription", ArrayType(StringType()), False),
                StructField("originalamount", ArrayType(StringType()), False),
                StructField("originalrate", ArrayType(StringType()), False)
            ]
        )
        @F.pandas_udf(schema, functionType=F.PandasUDFType.GROUPED_MAP)
        def inputevents_unit(df):
            patient_id = str(df["subject_id"].iloc[0])
            visit_id = str(df[self.visit_key].iloc[0])
            code = df["itemid"].tolist()
            name = ["itemid"] * len(code)
            timestamp = [str(t) for t in df["starttime"].tolist()]
            amount = df["amount"].tolist()
            amountuom = df["amountuom"].tolist()
            rate = df["rate"].tolist()
            rateuom = df["rateuom"].tolist()
            orderid = df["orderid"].tolist()
            linkorderid = df["linkorderid"].tolist()
            ordercategoryname = df["ordercategoryname"].tolist()
            secondaryordercategoryname = df["secondaryordercategoryname"].tolist()
            ordercomponenttypedescription = df["ordercomponenttypedescription"].tolist()
            ordercategorydescription = df["ordercategorydescription"].tolist()
            patientweight = df["patientweight"].tolist()
            totalamount = df["totalamount"].tolist()
            totalamountuom = df["totalamountuom"].tolist()
            isopenbag = df["isopenbag"].tolist()
            continueinnextdept = df["continueinnextdept"].tolist()
            statusdescription = df["statusdescription"].tolist()
            originalamount = df["originalamount"].tolist()
            originalrate = df["originalrate"].tolist()
            return pd.DataFrame([[
                patient_id,
                visit_id,
                code,
                name,
                timestamp,
                amount,
                amountuom,
                rate,
                rateuom,
                orderid,
                linkorderid,
                ordercategoryname,
                secondaryordercategoryname,
                ordercomponenttypedescription,
                ordercategorydescription,
                patientweight,
                totalamount,
                totalamountuom,
                isopenbag,
                continueinnextdept,
                statusdescription,
                originalamount,
                originalrate
            ]])
        
        pandas_df = spark_df.groupBy(
            "subject_id", self.visit_key
        ).apply(inputevents_unit).toPandas()
        
        def aggregate_events(row):
            events = []
            p_id = row["patient_id"]
            v_id = row["visit_id"]
            for (
                code, name, timestamp, amount, amountuom, rate, rateuom, orderid, linkorderid,
                ordercategoryname, secondaryordercategoryname, ordercomponenttypedescription,
                ordercategorydescription, patientweight, totalamount, totalamountuom, isopenbag,
                continueinnextdept, statusdescription, originalamount, originalrate
            ) in zip(
                row["code"], row["name"], row["timestamp"], row["amount"], row["amountuom"],
                row["rate"], row["rateuom"], row["orderid"], row["linkorderid"],
                row["ordercategoryname"], row["secondaryordercategoryname"],
                row["ordercomponenttypedescription"], row["ordercategorydescription"],
                row["patientweight"], row["totalamount"], row["totalamountuom"], row["isopenbag"],
                row["continueinnextdept"], row["statusdescription"], row["originalamount"],
                row["originalrate"]
            ):
                event = Event(
                    code=code,
                    name=name,
                    table=table,
                    vocabulary="MIMIC4_ITEMID",
                    visit_id=v_id,
                    patient_id=p_id,
                    timestamp=strptime(timestamp),
                    amount=amount,
                    amountuom=amountuom,
                    rate=rate,
                    rateuom=rateuom,
                    orderid=orderid,
                    linkorderid=linkorderid,
                    ordercategoryname=ordercategoryname,
                    secondaryordercategoryname=secondaryordercategoryname,
                    ordercomponenttypedescription=ordercomponenttypedescription,
                    ordercategorydescription=ordercategorydescription,
                    patientweight=patientweight,
                    totalamount=totalamount,
                    totalamountuom=totalamountuom,
                    isopenbag=isopenbag,
                    continueinnextdept=continueinnextdept,
                    statusdescription=statusdescription,
                    originalamount=originalamount,
                    originalrate=originalrate
                )
                events.append(event)
            return events
        
        # parallel apply to aggregate events
        aggregated_events = pandas_df.parallel_apply(aggregate_events, axis=1)
        
        # summarize the results
        patients = self._add_events_to_patient_dict(patients, aggregated_events)
        return patients

    def parse_chartevents(
        self,
        patients: Dict[str, Patient],
        spark: SparkSession,
    ) -> Dict[str, Patient]:
        """Helper function which parses chartevents table.
        
        Will be called in `self.parse_tables()`.
        
        Docs:
            - chartevents: https://mimic.mit.edu/docs/iv/modules/icu/chartevents/
        
        Returns:
            The updated patients dict.
        """
        table = "chartevents"
        
        # read tables
        spark_df = spark.read.csv(os.path.join(self.root, f"{table}.csv"), header=True)
        if self.dev:
            spark_df = spark_df.limit(3000)
        # convert dtype of charttime to datetime
        spark_df = spark_df.withColumn("charttime", F.to_timestamp("charttime"))
        
        if self.visit_unit == "icu":
            spark_df = self.filter_events(
                spark=spark,
                spark_df=spark_df,
                timestamp_key="charttime",
                joined_table="icustays",
                select=["stay_id", "intime", "outtime"],
                on="stay_id",
                should_infer_icustay_id=False
            )
        else:
            spark_df = self.filter_events(
                spark=spark,
                spark_df=spark_df,
                timestamp_key="charttime",
                joined_table="admissions",
                select=["hadm_id", "admittime", "dischtime"],
                on="hadm_id",
                should_infer_icustay_id=False
            )
        
        # sort by charttime
        spark_df = spark_df.sort(["subject_id", self.visit_key, "charttime"], ascending=True)
        
        schema = StructType(
            [
                StructField("patient_id", StringType(), False),
                StructField("visit_id", StringType(), False),
                StructField("code", ArrayType(StringType()), False),
                StructField("name", ArrayType(StringType()), False),
                StructField("timestamp", ArrayType(StringType()), False),
                StructField("value", ArrayType(StringType()), False),
                StructField("valuenum", ArrayType(StringType()), False),
                StructField("valueuom", ArrayType(StringType()), False),
                StructField("warning", ArrayType(StringType()), False),
            ]
        )
        @F.pandas_udf(schema, functionType=F.PandasUDFType.GROUPED_MAP)
        def chartevents_unit(df):
            patient_id = str(df["subject_id"].iloc[0])
            visit_id = str(df[self.visit_key].iloc[0])
            code = df["itemid"].tolist()
            name = ["itemid"] * len(code)
            timestamp = [str(t) for t in df["charttime"].tolist()]
            value = df["value"].tolist()
            valuenum = df["valuenum"].tolist()
            valueuom = df["valueuom"].tolist()
            warning = df["warning"].tolist()
            return pd.DataFrame([[
                patient_id,
                visit_id,
                code,
                name,
                timestamp,
                value,
                valuenum,
                valueuom,
                warning
            ]])
        
        pandas_df = spark_df.groupBy(
            "subject_id", self.visit_key
        ).apply(chartevents_unit).toPandas()
        
        def aggregate_events(row):
            events = []
            p_id = row["patient_id"]
            v_id = row["visit_id"]
            for code, name, timestamp, value, valuenum, valueuom, warning in zip(
                row["code"], row["name"], row["timestamp"], row["value"], row["valuenum"],
                row["valueuom"], row["warning"]
            ):
                event = Event(
                    code=code,
                    name=name,
                    table=table,
                    vocabulary="MIMIC4_ITEMID",
                    visit_id=v_id,
                    patient_id=p_id,
                    timestamp=strptime(timestamp),
                    value=value,
                    valuenum=valuenum,
                    valueuom=valueuom,
                    warning=warning
                )
                events.append(event)
            return events
        
        # parallel apply to aggregate events
        aggregated_events = pandas_df.parallel_apply(aggregate_events, axis=1)
        
        # summarize the results
        patients = self._add_events_to_patient_dict(patients, aggregated_events)
        return patients
    
    def parse_outputevents(
        self,
        patients: Dict[str, Patient],
        spark: SparkSession,
    ) -> Dict[str, Patient]:
        """Helper function which parses outputevents table.
        
        Docs:
            - outputevents: https://mimic.mit.edu/docs/iv/modules/icu/outputevents/
        
        Args:
            patients: a dict of `Patient` objects indexed by patient_id.
            spark: a spark session for reading the table.
        
        Returns:
            The updated patients dict.
        """
        table = "outputevents"
        
        # read tables
        spark_df = spark.read.csv(os.path.join(self.root, f"{table}.csv"), header=True)
        if self.dev:
            spark_df = spark_df.limit(3000)
        # convert dtype of charttime to datetime
        spark_df = spark_df.withColumn("charttime", F.to_timestamp("charttime"))
        
        if self.visit_unit == "icu":
            spark_df = self.filter_events(
                spark=spark,
                spark_df=spark_df,
                timestamp_key="charttime",
                joined_table="icustays",
                select=["stay_id", "intime", "outtime"],
                on="stay_id",
                should_infer_icustay_id=False
            )
        else:
            spark_df = self.filter_events(
                spark=spark,
                spark_df=spark_df,
                timestamp_key="charttime",
                joined_table="admissions",
                select=["hadm_id", "admittime", "dischtime"],
                on="hadm_id",
                should_infer_icustay_id=False
            )
        
        # sort by charttime
        spark_df = spark_df.sort(["subject_id", self.visit_key, "charttime"], ascending=True)
        
        schema = StructType(
            [
                StructField("patient_id", StringType(), False),
                StructField("visit_id", StringType(), False),
                StructField("code", ArrayType(StringType()), False),
                StructField("name", ArrayType(StringType()), False),
                StructField("timestamp", ArrayType(StringType()), False),
                StructField("value", ArrayType(StringType()), False),
                StructField("valueuom", ArrayType(StringType()), False),
            ]
        )
        @F.pandas_udf(schema, functionType=F.PandasUDFType.GROUPED_MAP)
        def outputevents_unit(df):
            patient_id = str(df["subject_id"].iloc[0])
            visit_id = str(df[self.visit_key].iloc[0])
            code = df["itemid"].tolist()
            name = ["itemid"] * len(code)
            timestamp = [str(t) for t in df["charttime"].tolist()]
            value = df["value"].tolist()
            valueuom = df["valueuom"].tolist()
            return pd.DataFrame([[
                patient_id,
                visit_id,
                code,
                name,
                timestamp,
                value,
                valueuom
            ]])
        
        pandas_df = spark_df.groupBy(
            "subject_id", self.visit_key
        ).apply(outputevents_unit).toPandas()

        def aggregate_events(row):
            events = []
            p_id = row["patient_id"]
            v_id = row["visit_id"]
            for code, name, timestamp, value, valueuom in zip(
                row["code"], row["name"], row["timestamp"], row["value"], row["valueuom"]
            ):
                event = Event(
                    code=code,
                    name=name,
                    table=table,
                    vocabulary="MIMIC4_ITEMID",
                    visit_id=v_id,
                    patient_id=p_id,
                    timestamp=strptime(timestamp),
                    value=value,
                    valueuom=valueuom
                )
                events.append(event)
            return events
        
        # parallel apply to aggregate events
        aggregated_events = pandas_df.parallel_apply(aggregate_events, axis=1)
        
        # summarize the results
        patients = self._add_events_to_patient_dict(patients, aggregated_events)
        return patients

    def parse_procedureevents(
        self,
        patients: Dict[str, Patient],
        spark: SparkSession
    ) -> Dict[str, Patient]:
        """Helper function which parses procedureevents table.
        
        Will be called in `self.parse_tables()`.
        
        Docs:
            - procedureevents: https://mimic.mit.edu/docs/iv/modules/icu/procedureevents/
        
        Args:
            patients: a dict of `Patient` objects indexed by patient_id.
            spark: a spark session for reading the table.
        
        Returns:
            The updated patients dict.
        """
        table = "procedureevents"
        
        # read tables
        spark_df = spark.read.csv(os.path.join(self.root, f"{table}.csv"), header=True)
        if self.dev:
            spark_df = spark_df.limit(3000)
        # convert dtype of charttime
        spark_df = spark_df.withColumn("starttime", F.to_timestamp("starttime"))
        
        if self.visit_unit == "icu":
            spark_df = self.filter_events(
                spark=spark,
                spark_df=spark_df,
                timestamp_key="starttime",
                joined_table="icustays",
                select=["stay_id", "intime", "outtime"],
                on="stay_id",
                should_infer_icustay_id=False
            )
        else:
            spark_df = self.filter_events(
                spark=spark,
                spark_df=spark_df,
                timestamp_key="starttime",
                joined_table="admissions",
                select=["hadm_id", "admittime", "dischtime"],
                on="hadm_id",
                should_infer_icustay_id=False
            )
        
        # sort by starttime
        spark_df = spark_df.sort(["subject_id", self.visit_key, "starttime"], ascending=True)
        
        schema = StructType(
            [
                StructField("patient_id", StringType(), False),
                StructField("visit_id", StringType(), False),
                StructField("code", ArrayType(StringType()), False),
                StructField("name", ArrayType(StringType()), False),
                StructField("timestamp", ArrayType(StringType()), False),
                StructField("value", ArrayType(StringType()), False),
                StructField("valueuom", ArrayType(StringType()), False),
                StructField("location", ArrayType(StringType()), False),
                StructField("locationcategory", ArrayType(StringType()), False),
                StructField("orderid", ArrayType(StringType()), False),
                StructField("linkorderid", ArrayType(StringType()), False),
                StructField("ordercategoryname", ArrayType(StringType()), False),
                StructField("ordercategorydescription", ArrayType(StringType()), False),
                StructField("patientweight", ArrayType(StringType()), False),
                StructField("isopenbag", ArrayType(StringType()), False),
                StructField("continueinnextdept", ArrayType(StringType()), False),
                StructField("statusdescription", ArrayType(StringType()), False),
                StructField("originalamount", ArrayType(StringType()), False),
                StructField("originalrate", ArrayType(StringType()), False),
            ]
        )
        @F.pandas_udf(schema, functionType=F.PandasUDFType.GROUPED_MAP)
        def procedureevents_unit(df):
            patient_id = str(df["subject_id"].iloc[0])
            visit_id = str(df[self.visit_key].iloc[0])
            code = df["itemid"].tolist()
            name = ["itemid"] * len(code)
            timestamp = [str(t) for t in df["starttime"].tolist()]
            value = df["value"].tolist()
            valueuom = df["valueuom"].tolist()
            location = df["location"].tolist()
            locationcategory = df["locationcategory"].tolist()
            orderid = df["orderid"].tolist()
            linkorderid = df["linkorderid"].tolist()
            ordercategoryname = df["ordercategoryname"].tolist()
            ordercategorydescription = df["ordercategorydescription"].tolist()
            patientweight = df["patientweight"].tolist()
            isopenbag = df["isopenbag"].tolist()
            continueinnextdept = df["continueinnextdept"].tolist()
            statusdescription = df["statusdescription"].tolist()
            originalamount = df["originalamount"].tolist()
            originalrate = df["originalrate"].tolist()
            return pd.DataFrame([[
                patient_id,
                visit_id,
                code,
                name,
                timestamp,
                value,
                valueuom,
                location,
                locationcategory,
                orderid,
                linkorderid,
                ordercategoryname,
                ordercategorydescription,
                patientweight,
                isopenbag,
                continueinnextdept,
                statusdescription,
                originalamount,
                originalrate
            ]])
            
        pandas_df = spark_df.groupBy(
            "subject_id", self.visit_key
        ).apply(procedureevents_unit).toPandas()
        
        def aggregate_events(row):
            events = []
            p_id = row["patient_id"]
            v_id = row["visit_id"]
            for (
                code, name, timestamp, value, valueuom, location, locationcategory, orderid,
                linkorderid, ordercategoryname, ordercategorydescription, patientweight, isopenbag,
                continueinnextdept, statusdescription, originalamount, originalrate
            ) in zip(
                row["code"], row["name"], row["timestamp"], row["value"], row["valueuom"],
                row["location"], row["locationcategory"], row["orderid"], row["linkorderid"],
                row["ordercategoryname"], row["ordercategorydescription"], row["patientweight"],
                row["isopenbag"], row["continueinnextdept"], row["statusdescription"],
                row["originalamount"], row["originalrate"]
            ):
                event = Event(
                    code=code,
                    name=name,
                    table=table,
                    vocabulary="MIMIC4_ITEMID",
                    visit_id=v_id,
                    patient_id=p_id,
                    timestamp=strptime(timestamp),
                    value=value,
                    valueuom=valueuom,
                    location=location,
                    locationcategory=locationcategory,
                    orderid=orderid,
                    linkorderid=linkorderid,
                    ordercategoryname=ordercategoryname,
                    ordercategorydescription=ordercategorydescription,
                    patientweight=patientweight,
                    isopenbag=isopenbag,
                    continueinnextdept=continueinnextdept,
                    statusdescription=statusdescription,
                    originalamount=originalamount,
                    originalrate=originalrate
                )
                events.append(event)
            return events

        # parallel apply to aggregate events
        aggregated_events = pandas_df.parallel_apply(aggregate_events, axis=1)
        
        # summarize the results
        patients = self._add_events_to_patient_dict(patients, aggregated_events)
        return patients
    
    def parse_microbiologyevents(
        self,
        patients: Dict[str, Patient],
        spark: SparkSession,
    ) -> Dict[str, Patient]:
        """Helper function which parses microbiologyevents table.
        
        Will be called in `self.parse_tables()`.
        
        Docs:
            - microbiologyevents: https://mimic.mit.edu/docs/iv/modules/hosp/microbiologyevents/
        
        Args:
            patients: a dict of `Patient` objects indexed by patient_id.
            spark: a spark session for reading the table.
        
        Returns:
            The updated patients dict.
        """
        table = "microbiologyevents"
        
        # read tables
        spark_df = spark.read.csv(os.path.join(self.root, f"{table}.csv"), header=True)
        if self.dev:
            spark_df = spark_df.limit(3000)
        # convert dtype of charttime to datetime
        spark_df = spark_df.withColumn("charttime", F.to_timestamp("charttime"))
        
        if self.visit_unit == "icu":
            spark_df = self.filter_events(
                spark=spark,
                spark_df=spark_df,
                timestamp_key="charttime",
                joined_table="icustays",
                select=["hadm_id", "stay_id", "intime", "outtime"],
                on="hadm_id",
                should_infer_icustay_id=True,
            )
        else:
            spark_df = self.filter_events(
                spark=spark,
                spark_df=spark_df,
                timestamp_key="charttime",
                joined_table="admissions",
                select=["hadm_id", "admittime", "dischtime"],
                on="hadm_id",
                should_infer_icustay_id=False,
            )
        # sort by charttime
        spark_df = spark_df.sort(["subject_id", self.visit_key, "charttime"], ascending=True)
        
        schema = StructType(
            [
                StructField("patient_id", StringType(), False),
                StructField("visit_id", StringType(), False),
                StructField("code", ArrayType(StringType()), False),
                StructField("name", ArrayType(StringType()), False),
                StructField("timestamp", ArrayType(StringType()), False),
                StructField("spec_itemid", ArrayType(StringType()), False),
                StructField("spec_type_desc", ArrayType(StringType()), False),
                StructField("test_seq", ArrayType(StringType()), False),
                StructField("test_itemid", ArrayType(StringType()), False),
                StructField("test_name", ArrayType(StringType()), False),
                StructField("org_itemid", ArrayType(StringType()), False),
                StructField("org_name", ArrayType(StringType()), False),
                StructField("isolate_num", ArrayType(StringType()), False),
                StructField("quantity", ArrayType(StringType()), False),
                StructField("ab_itemid", ArrayType(StringType()), False),
                StructField("ab_name", ArrayType(StringType()), False),
                StructField("dilution_text", ArrayType(StringType()), False),
                StructField("dilution_comparison", ArrayType(StringType()), False),
                StructField("dilution_value", ArrayType(StringType()), False),
                StructField("interpretation", ArrayType(StringType()), False),
                StructField("comments", ArrayType(StringType()), False),
            ]
        )
        @F.pandas_udf(schema, functionType=F.PandasUDFType.GROUPED_MAP)
        def microbiologyevents_unit(df):
            patient_id = str(df["subject_id"].iloc[0])
            visit_id = str(df[self.visit_key].iloc[0])
            code = [None] * len(df)
            name = [None] * len(code)
            timestamp = [str(t) for t in df["charttime"].tolist()]
            spec_itemid = df["spec_itemid"].tolist()
            spec_type_desc = df["spec_type_desc"].tolist()
            test_seq = df["test_seq"].tolist()
            test_itemid = df["test_itemid"].tolist()
            test_name = df["test_name"].tolist()
            org_itemid = df["org_itemid"].tolist()
            org_name = df["org_name"].tolist()
            isolate_num = df["isolate_num"].tolist()
            quantity = df["quantity"].tolist()
            ab_itemid = df["ab_itemid"].tolist()
            ab_name = df["ab_name"].tolist()
            dilution_text = df["dilution_text"].tolist()
            dilution_comparison = df["dilution_comparison"].tolist()
            dilution_value = df["dilution_value"].tolist()
            interpretation = df["interpretation"].tolist()
            comments = df["comments"].tolist()
            return pd.DataFrame([[
                patient_id,
                visit_id,
                code,
                name,
                timestamp,
                spec_itemid,
                spec_type_desc,
                test_seq,
                test_itemid,
                test_name,
                org_itemid,
                org_name,
                isolate_num,
                quantity,
                ab_itemid,
                ab_name,
                dilution_text,
                dilution_comparison,
                dilution_value,
                interpretation,
                comments
            ]])
        
        pandas_df = spark_df.groupBy(
            "subject_id", self.visit_key
        ).apply(microbiologyevents_unit).toPandas()
        
        def aggregate_events(row):
            events = []
            p_id = row["patient_id"]
            v_id = row["visit_id"]
            for (
                code, name, timestamp, spec_itemid, spec_type_desc, test_seq, test_itemid,
                test_name, org_itemid, org_name, isolate_num, quantity, ab_itemid, ab_name,
                dilution_text, dilution_comparison, dilution_value, interpretation, comments
            ) in zip(
                row["code"], row["name"], row["timestamp"], row["spec_itemid"],
                row["spec_type_desc"], row["test_seq"], row["test_itemid"], row["test_name"],
                row["org_itemid"], row["org_name"], row["isolate_num"], row["quantity"],
                row["ab_itemid"], row["ab_name"], row["dilution_text"], row["dilution_comparison"],
                row["dilution_value"], row["interpretation"], row["comments"]
            ):
                event = Event(
                    code=code,
                    name=name,
                    table=table,
                    vocabulary="MIMIC4_ITEMID",
                    visit_id=v_id,
                    patient_id=p_id,
                    timestamp=strptime(timestamp),
                    spec_itemid=spec_itemid,
                    spec_type_desc=spec_type_desc,
                    test_seq=test_seq,
                    test_itemid=test_itemid,
                    test_name=test_name,
                    org_itemid=org_itemid,
                    org_name=org_name,
                    isolate_num=isolate_num,
                    quantity=quantity,
                    ab_itemid=ab_itemid,
                    ab_name=ab_name,
                    dilution_text=dilution_text,
                    dilution_comparison=dilution_comparison,
                    dilution_value=dilution_value,
                    interpretation=interpretation,
                    comments=comments
                )
                events.append(event)
            return events

        # parallel apply to aggregate events
        aggregated_events = pandas_df.parallel_apply(aggregate_events, axis=1)
        
        # summarize the results
        patients = self._add_events_to_patient_dict(patients, aggregated_events)
        return patients
    

if __name__ == "__main__":
    dataset = MIMIC4Dataset(
        root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",
        tables=["diagnoses_icd", "procedures_icd", "prescriptions", "labevents", "hcpcsevents"],
        code_mapping={"NDC": "ATC"},
        refresh_cache=False,
    )
    dataset.stat()
    dataset.info()
