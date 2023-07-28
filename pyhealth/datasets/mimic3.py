import os
from typing import Optional, List, Dict, Tuple, Union

import pandas as pd
import datetime

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, ArrayType, StringType

from pyhealth.data import Event, Visit, Patient
from pyhealth.datasets import BaseEHRDataset, BaseEHRSparkDataset
from pyhealth.datasets.utils import strptime

# TODO: add other tables


class MIMIC3Dataset(BaseEHRDataset):
    """Base dataset for MIMIC-III dataset.

    The MIMIC-III dataset is a large dataset of de-identified health records of ICU
    patients. The dataset is available at https://mimic.physionet.org/.

    The basic information is stored in the following tables:
        - PATIENTS: defines a patient in the database, SUBJECT_ID.
        - ADMISSIONS: defines a patient's hospital admission, HADM_ID.

    We further support the following tables:
        - DIAGNOSES_ICD: contains ICD-9 diagnoses (ICD9CM code) for patients.
        - PROCEDURES_ICD: contains ICD-9 procedures (ICD9PROC code) for patients.
        - PRESCRIPTIONS: contains medication related order entries (NDC code)
            for patients.
        - LABEVENTS: contains laboratory measurements (MIMIC3_ITEMID code)
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
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> dataset = MIMIC3Dataset(
        ...         root="/srv/local/data/physionet.org/files/mimiciii/1.4",
        ...         tables=["DIAGNOSES_ICD", "PRESCRIPTIONS"],
        ...         code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
        ...     )
        >>> dataset.stat()
        >>> dataset.info()
    """

    def parse_basic_info(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses PATIENTS and ADMISSIONS tables.

        Will be called in `self.parse_tables()`

        Docs:
            - PATIENTS: https://mimic.mit.edu/docs/iii/tables/patients/
            - ADMISSIONS: https://mimic.mit.edu/docs/iii/tables/admissions/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id which is updated with the mimic-3 table result.

        Returns:
            The updated patients dict.
        """
        # read patients table
        patients_df = pd.read_csv(
            os.path.join(self.root, "PATIENTS.csv"),
            dtype={"SUBJECT_ID": str},
            nrows=1000 if self.dev else None,
        )
        # read admissions table
        admissions_df = pd.read_csv(
            os.path.join(self.root, "ADMISSIONS.csv"),
            dtype={"SUBJECT_ID": str, "HADM_ID": str},
        )
        # merge patient and admission tables
        df = pd.merge(patients_df, admissions_df, on="SUBJECT_ID", how="inner")
        # sort by admission and discharge time
        df = df.sort_values(["SUBJECT_ID", "ADMITTIME", "DISCHTIME"], ascending=True)
        # group by patient
        df_group = df.groupby("SUBJECT_ID")

        # parallel unit of basic information (per patient)
        def basic_unit(p_id, p_info):
            patient = Patient(
                patient_id=p_id,
                birth_datetime=strptime(p_info["DOB"].values[0]),
                death_datetime=strptime(p_info["DOD_HOSP"].values[0]),
                gender=p_info["GENDER"].values[0],
                ethnicity=p_info["ETHNICITY"].values[0],
            )
            # load visits
            for v_id, v_info in p_info.groupby("HADM_ID"):
                visit = Visit(
                    visit_id=v_id,
                    patient_id=p_id,
                    encounter_time=strptime(v_info["ADMITTIME"].values[0]),
                    discharge_time=strptime(v_info["DISCHTIME"].values[0]),
                    discharge_status=v_info["HOSPITAL_EXPIRE_FLAG"].values[0],
                )
                # add visit
                patient.add_visit(visit)
            return patient

        # parallel apply
        df_group = df_group.parallel_apply(
            lambda x: basic_unit(x.SUBJECT_ID.unique()[0], x)
        )
        # summarize the results
        for pat_id, pat in df_group.items():
            patients[pat_id] = pat

        return patients

    def parse_diagnoses_icd(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses DIAGNOSES_ICD table.

        Will be called in `self.parse_tables()`

        Docs:
            - DIAGNOSES_ICD: https://mimic.mit.edu/docs/iii/tables/diagnoses_icd/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.

        Note:
            MIMIC-III does not provide specific timestamps in DIAGNOSES_ICD
                table, so we set it to None.
        """
        table = "DIAGNOSES_ICD"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"SUBJECT_ID": str, "HADM_ID": str, "ICD9_CODE": str},
        )
        # drop records of the other patients
        df = df[df["SUBJECT_ID"].isin(patients.keys())]
        # drop rows with missing values
        df = df.dropna(subset=["SUBJECT_ID", "HADM_ID", "ICD9_CODE"])
        # sort by sequence number (i.e., priority)
        df = df.sort_values(["SUBJECT_ID", "HADM_ID", "SEQ_NUM"], ascending=True)
        # group by patient and visit
        group_df = df.groupby("SUBJECT_ID")

        # parallel unit of diagnosis (per patient)
        def diagnosis_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("HADM_ID"):
                for code in v_info["ICD9_CODE"]:
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary="ICD9CM",
                        visit_id=v_id,
                        patient_id=p_id,
                    )
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: diagnosis_unit(x.SUBJECT_ID.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def parse_procedures_icd(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses PROCEDURES_ICD table.

        Will be called in `self.parse_tables()`

        Docs:
            - PROCEDURES_ICD: https://mimic.mit.edu/docs/iii/tables/procedures_icd/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.

        Note:
            MIMIC-III does not provide specific timestamps in PROCEDURES_ICD
                table, so we set it to None.
        """
        table = "PROCEDURES_ICD"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"SUBJECT_ID": str, "HADM_ID": str, "ICD9_CODE": str},
        )
        # drop records of the other patients
        df = df[df["SUBJECT_ID"].isin(patients.keys())]
        # drop rows with missing values
        df = df.dropna(subset=["SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD9_CODE"])
        # sort by sequence number (i.e., priority)
        df = df.sort_values(["SUBJECT_ID", "HADM_ID", "SEQ_NUM"], ascending=True)
        # group by patient and visit
        group_df = df.groupby("SUBJECT_ID")

        # parallel unit of procedure (per patient)
        def procedure_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("HADM_ID"):
                for code in v_info["ICD9_CODE"]:
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary="ICD9PROC",
                        visit_id=v_id,
                        patient_id=p_id,
                    )
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: procedure_unit(x.SUBJECT_ID.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def parse_prescriptions(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses PRESCRIPTIONS table.

        Will be called in `self.parse_tables()`

        Docs:
            - PRESCRIPTIONS: https://mimic.mit.edu/docs/iii/tables/prescriptions/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "PRESCRIPTIONS"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            low_memory=False,
            dtype={"SUBJECT_ID": str, "HADM_ID": str, "NDC": str},
        )
        # drop records of the other patients
        df = df[df["SUBJECT_ID"].isin(patients.keys())]
        # drop rows with missing values
        df = df.dropna(subset=["SUBJECT_ID", "HADM_ID", "NDC"])
        # sort by start date and end date
        df = df.sort_values(
            ["SUBJECT_ID", "HADM_ID", "STARTDATE", "ENDDATE"], ascending=True
        )
        # group by patient and visit
        group_df = df.groupby("SUBJECT_ID")

        # parallel unit for prescription (per patient)
        def prescription_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("HADM_ID"):
                for timestamp, code in zip(v_info["STARTDATE"], v_info["NDC"]):
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary="NDC",
                        visit_id=v_id,
                        patient_id=p_id,
                        timestamp=strptime(timestamp),
                    )
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: prescription_unit(x.SUBJECT_ID.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

    def parse_labevents(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses LABEVENTS table.

        Will be called in `self.parse_tables()`

        Docs:
            - LABEVENTS: https://mimic.mit.edu/docs/iii/tables/labevents/

        Args:
            patients: a dict of `Patient` objects indexed by patient_id.

        Returns:
            The updated patients dict.
        """
        table = "LABEVENTS"
        # read table
        df = pd.read_csv(
            os.path.join(self.root, f"{table}.csv"),
            dtype={"SUBJECT_ID": str, "HADM_ID": str, "ITEMID": str},
        )
        # drop records of the other patients
        df = df[df["SUBJECT_ID"].isin(patients.keys())]
        # drop rows with missing values
        df = df.dropna(subset=["SUBJECT_ID", "HADM_ID", "ITEMID"])
        # sort by charttime
        df = df.sort_values(["SUBJECT_ID", "HADM_ID", "CHARTTIME"], ascending=True)
        # group by patient and visit
        group_df = df.groupby("SUBJECT_ID")

        # parallel unit for lab (per patient)
        def lab_unit(p_id, p_info):
            events = []
            for v_id, v_info in p_info.groupby("HADM_ID"):
                for timestamp, code in zip(v_info["CHARTTIME"], v_info["ITEMID"]):
                    event = Event(
                        code=code,
                        table=table,
                        vocabulary="MIMIC3_ITEMID",
                        visit_id=v_id,
                        patient_id=p_id,
                        timestamp=strptime(timestamp),
                    )
                    events.append(event)
            return events

        # parallel apply
        group_df = group_df.parallel_apply(
            lambda x: lab_unit(x.SUBJECT_ID.unique()[0], x)
        )

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, group_df)
        return patients

class MIMIC3SparkDataset(BaseEHRSparkDataset):
    """Base dataset for MIMIC-III dataset utilized by PySpark for the efficient data processing.
    
    The MIMIC-III dataset is a large dataset of de-identified health records of ICU
    patients. The dataset is available at https://mimic.physionet.org/.

    Unlike the normal mimic3 dataset (MIMIC3Dataset) that does not utilize PySpark, this
    dataset provides not only the corresponding code for each event but also any other
    available columns by attr_dict or internal attributes to support further data processing.

    The basic information is stored in the following tables:
        - PATIENTS: defines a patient in the database, SUBJECT_ID.
        - ADMISSIONS: defines a patient's hospital admission, HADM_ID.
        - ICUSTAYS: defines a patient's ICU stay, ICUSTAY_ID.
    
    We further support the following tables:
        - DIAGNOSES_ICD: contains ICD-9 diagnoses (ICD9CM code) for patients.
        - PROCEDURES_ICD: contains ICD-9 procedures (ICD9PROC code) for patients.
        - PRESCRIPTIONS: contains medication related order entries (NCD code) for patients.
        - LABEVENTS: contains laboratory measurements (MIMIC3_ITEMID code) for patients.
        - CHARTEVENTS: contains all charted data (MIMIC3_ITEMID code) for patients.
        - MICROBIOLOGYEVENTS: contains microbiology information for patients, including cultures
            acquired and associated sensitivities.
        - INPUTEVENTS_CV: contains input data (MIMIC3_ITEMID code) for patients, stored in CareVue
            ICU databases.
        - INPUTEVENTS_MV: contains input data (MIMIC3_ITEMID code) for patients, stored in
            Metavision ICU databases.
        - OUTPUTEVENTS: contains output data (MIMIC3_ITEMID code) for patients, stored in CareVue
            or Metavision ICU databases.
        - PROCEDUREEVENTS_MV: contains procedures (MIMIC3_ITEMID code) for patients, stored in
            Metavision ICU databases.
    
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
        code_mapping: a dictionary containing the code mapping information.
            The key is a str of the source code vocabulary and the value is of
            two formats:
                - a str of the target code vocabulary. E.g., {"NDC", "ATC"}.
                - a tuple with two elements. The first element is a str of the
                    target code vocabulary and the second element is a dict with
                    keys "source_kwargs" or "target_kwargs" and values of the
                    corresponding kwargs for the `CrossMap.map()` method. E.g.,
                    {"NDC", ("ATC", {"target_kwargs": {"level": 3}})}.
                Default is empty dict, which means the original code will be used.
        dev: whether to enable dev mode (only use a small subset of the data).
            Default is False.
        refresh_cache: whether to refresh the cache; if true, the dataset will
            be processed from scratch and the cache will be updated. Default is False.

    Examples:
        >>> from pyhealth.datasets import MIMIC3SparkDataset
        >>> dataset = MIMIC3SparkDataset(
        ...     root="/usr/local/data/physionet.org/files/mimiciii/1,4",
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
            self.visit_key = "HADM_ID"
            self.encounter_key = "ADMITTIME"
            self.discharge_key = "DISCHTIME"
        elif visit_unit == "icu":
            self.visit_key = "ICUSTAY_ID"
            self.encounter_key = "INTIME"
            self.discharge_key = "OUTTIME"
        super().__init__(root, tables, visit_unit, **kwargs)

    def parse_basic_info(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """Helper function which parses PATIETNS, ADMISSIONS, ICUSTAYS, and DIAGNOSES_ICD tables.
        
        Will be called in `self.parse_tables()`.
        
        Docs:
            - PATIENTS: https://mimic.mit.edu/docs/iii/tables/patients/
            - ADMISSIONS: https://mimic.mit.edu/docs/iii/tables/admissions/
            - ICUSTAYS: https://mimic.mit.edu/docs/iii/tables/icustays/
            - DIAGNOSES_ICD: https://mimic.mit.edu/docs/iii/tables/diagnoses_icd/
        
        Args:
            patients: a dict of `Patient` objects indexed by patient_id which is updated with the
                mimic-iii table result.
        
        Returns:
            The updated patients dict.
        """
        # read patients table
        patients_df = pd.read_csv(
            os.path.join(self.root, "PATIENTS.csv"),
            dtype={"SUBJECT_ID": str},
            nrows=1000 if self.dev else None,
        )
        # read admissions table
        admissions_df = pd.read_csv(
            os.path.join(self.root, "ADMISSIONS.csv"),
            dtype={"SUBJECT_ID": str, "HADM_ID": str},
        )
        # read icustays table
        icustays_df = pd.read_csv(
            os.path.join(self.root, "ICUSTAYS.csv"),
            dtype={"HADM_ID": str, "ICUSTAY_ID": str}
        )
        # read diagnoses table
        diagnoses_df = pd.read_csv(
            os.path.join(self.root, "DIAGNOSES_ICD.csv"),
            dtype={"HADM_ID": str}
        )
        diagnoses_df = diagnoses_df.groupby(
            "HADM_ID"
        )["ICD9_CODE"].agg(lambda x: list(set(x))).to_frame()

        # merge patient, admission, icustay tables
        df = pd.merge(
            patients_df,
            admissions_df,
            on="SUBJECT_ID",
            how="inner"
        )
        df = pd.merge(
            df,
            icustays_df[["HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME"]],
            on="HADM_ID",
            how="inner"
        )
        # merge with diagnoses_icd table
        df = pd.merge(
            df,
            diagnoses_df,
            on="HADM_ID",
            how="inner"
        )
        # sort by admission and discharge time
        df = df.sort_values(["SUBJECT_ID", self.encounter_key, self.discharge_key], ascending=True)
        # group by patient
        df_group = df.groupby("SUBJECT_ID")

        # parallel unit of basic information (per patient)
        def basic_unit(p_id, p_info):
            patient = Patient(
                patient_id=p_id,
                birth_datetime=strptime(p_info["DOB"].values[0]),
                death_datetime=strptime(p_info["DOD_HOSP"].values[0]),
                gender=p_info["GENDER"].values[0],
                ethnicity=p_info["ETHNICITY"].values[0],
            )
            # load visits
            for v_id, v_info in p_info.groupby(self.visit_key):
                visit = Visit(
                    visit_id=v_id,
                    patient_id=p_id,
                    encounter_time=strptime(v_info[self.encounter_key].values[0]),
                    discharge_time=strptime(v_info[self.discharge_key].values[0]),
                    discharge_status=v_info["HOSPITAL_EXPIRE_FLAG"].values[0],
                    discharge_location=v_info["DISCHARGE_LOCATION"].values[0],
                    hadm_id=v_info["HADM_ID"].values[0],
                    hospital_discharge_time=strptime(v_info["DISCHTIME"].values[0]),
                    diagnosis_codes=v_info["ICD9_CODE"].values[0]
                )
                # add visit
                patient.add_visit(visit)
            return patient

        # parallel apply to aggregate events
        df_group = df_group.parallel_apply(
            lambda x: basic_unit(x.SUBJECT_ID.unique()[0], x)
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
        """Helper function which parses DIAGNOSES_ICD table.
        
        Will be called in `self.parse_tables()`.
        
        Docs:
            - DIAGNOSES_ICD: https://mimic.mit.edu/docs/iii/tables/diagnoses_icd/
        
        Args:
            patients: a dict of `Patient` objects indexed by patient_id.
            spark: a spark session for reading the table.
        
        Returns:
            The updated patients dicrt.
        
        Note:
            MIMIC-III does not provide specific timestamps in DIAGNOSES_ICD
                table, so we set it to "the end of the hospital stay".
        """
        table = "DIAGNOSES_ICD"
        
        # read table
        spark_df = spark.read.csv(os.path.join(self.root, f"{table}.csv"), header=True)
        if self.dev:
            spark_df = spark_df.limit(3000)
        
        if self.visit_unit == "icu":
            raise ValueError(
                "Cannot determine icu-level diagnosis events since the ICD codes are generated at "
                "the end of the hospital stay. Please exclude DIAGNOSES_ICD table or instantiate "
                ":MIMIC3SparkDataset: class with hospital-level by specifying "
                "`visit_unit='hospital'`."
            )
        else:
            spark_df = self.filter_events(
                spark=spark,
                spark_df=spark_df,
                timestamp_key=None,
                table=table,
                joined_table="ADMISSIONS",
                select=["HADM_ID", "ADMITTIME", "DISCHTIME"],
                on="HADM_ID",
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
            ]
        )
        @F.pandas_udf(schema, functionType=F.PandasUDFType.GROUPED_MAP)
        def diagnoses_icd_unit(df):
            patient_id = str(df["SUBJECT_ID"].iloc[0])
            visit_id = str(df[self.visit_key].iloc[0])
            code = df["ICD9_CODE"].tolist()
            name = ["ICD9_CODE"] * len(code)
            # NOTE: diagnosis codes are generated at the end of the hospital stay.
            timestamp = [str(t) for t in df["DISCHTIME"].tolist()]
            seq_num = df["SEQ_NUM"].tolist()
            return pd.DataFrame([[
                patient_id,
                visit_id,
                code,
                name,
                timestamp,
                seq_num
            ]])
        
        pandas_df = spark_df.groupBy(
            "SUBJECT_ID", self.visit_key
        ).apply(diagnoses_icd_unit).toPandas()
        
        def aggregate_events(row):
            events = []
            p_id = row["patient_id"]
            v_id = row["visit_id"]
            for code, name, timestamp, seq_num in zip(
                row["code"], row["name"], row["timestamp"], row["seq_num"]
            ):
                event = Event(
                    code=code,
                    name=name,
                    table=table,
                    vocabulary="ICD9CM",
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
        
    def parse_procedures_icd(
        self,
        patients: Dict[str, Patient],
        spark: SparkSession
    ) -> Dict[str, Patient]:
        """Helper function which parses PROCEDURES_ICD table.
        
        Will be called in `self.parse_tables()`.
        
        Docs:
            - PROCEDURES_ICD: https://mimic.mit.edu/docs/iii/tables/procedures_icd/
        
        Args:
            patients: a dict of `Patient` objects indexed by patient_id.
            spark: a spark session for reading the table.

        Returns:
            The updated patients dict.
        
        Note:
            MIMIC-III does not provide specific timestamps in PROCEDURES_ICD
                table, so we set it to "the end of the hospital stay".
        """
        table = "PROCEDURES_ICD"
        
        # read table
        spark_df = spark.read.csv(os.path.join(self.root, f"{table}.csv"), header=True)
        if self.dev:
            spark_df = spark_df.limit(3000)
        
        if self.visit_unit == "icu":
            raise ValueError(
                "Cannot determine icu-level procedure events since the ICD codes are generated at "
                "the end of the hospital stay. Please exclude PROCEDURES_ICD table or instantiate "
                "MIMIC3SparkDataset class with hospital-level by specifying "
                "`visit_unit='hospital'`."
            )
        else:
            spark_df = self.filter_events(
                spark=spark,
                spark_df=spark_df,
                timestamp_key=None,
                table=table,
                joined_table="ADMISSIONS",
                on="HADM_ID",
                select=["HADM_ID", "ADMITTIME", "DISCHTIME"]
            )
        
        schema = StructType(
            [
                StructField("patient_id", StringType(), False),
                StructField("visit_id", StringType(), False),
                StructField("code", ArrayType(StringType()), False),
                StructField("name", ArrayType(StringType()), False),
                StructField("timestamp", ArrayType(StringType()), False),
                StructField("seq_num", ArrayType(StringType()), False),
            ]
        )
        @F.pandas_udf(schema, functionType=F.PandasUDFType.GROUPED_MAP)
        def procedures_icd_unit(df):
            patient_id = str(df["SUBJECT_ID"].iloc[0])
            visit_id = str(df[self.visit_key].iloc[0])
            code = df["ICD9_CODE"].tolist()
            name = ["ICD9_CODE"] * len(code)
            # NOTE: diagnosis codes are generated at the end of the hospital stay.
            timestamp = [str(t) for t in df["DISCHTIME"].tolist()]
            seq_num = df["SEQ_NUM"].tolist()
            return pd.DataFrame([[
                patient_id,
                visit_id,
                code,
                name,
                timestamp,
                seq_num
            ]])
        
        pandas_df = spark_df.groupBy(
            "SUBJECT_ID", self.visit_key
        ).apply(procedures_icd_unit).toPandas()
        
        def aggregate_events(row):
            events = []
            p_id = row["patient_id"]
            v_id = row["visit_id"]
            for code, name, timestamp, seq_num in zip(
                row["code"], row["name"], row["timestamp"], row["seq_num"]
            ):
                event = Event(
                    code=code,
                    name=name,
                    table=table,
                    vocabulary="ICD9PROC",
                    visit_id=v_id,
                    patient_id=p_id,
                    timestamp=strptime(timestamp),
                    seq_num=seq_num
                )
                events.append(event)
            return events

        # parallel applly to aggregate events
        aggregated_events = pandas_df.parallel_apply(aggregate_events, axis=1)

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, aggregated_events)
        return patients

    def parse_prescriptions(
        self,
        patients: Dict[str, Patient],
        spark: SparkSession,
    ) -> Dict[str, Patient]:
        """Helper function which parses PRESCRIPTIONS table.
        
        Will be called in `self.parse_tables()`.
        
        Docs:
            - PRESCRIPTIONS: https://mimic.mit.edu/docs/iii/tables/prescriptions/
        
        Args:
            patients: a dict of `Patient` objects indexed by patient_id.
            spark: a spark session for reading the table.
        
        Returns:
            The updated patients dict.
        """
        table = "PRESCRIPTIONS"

        spark_df = spark.read.csv(os.path.join(self.root, f"{table}.csv"), header=True)
        if self.dev:
            spark_df = spark_df.limit(3000)
        # convert dtype of startdate to datetime
        spark_df = spark_df.withColumn("STARTDATE", F.to_timestamp("STARTDATE"))

        if self.visit_unit == "icu":
            spark_df = self.filter_events(
                spark=spark,
                spark_df=spark_df,
                timestamp_key="STARTDATE",
                joined_table="ICUSTAYS",
                select=["ICUSTAY_ID", "INTIME", "OUTTIME"],
                on="ICUSTAY_ID",
                should_infer_icustay_id=False,
            )
        else:
            spark_df = self.filter_events(
                spark=spark,
                spark_df=spark_df,
                timestamp_key="STARTDATE",
                joined_table="ADMISSIONS",
                select=["HADM_ID", "ADMITTIME", "DISCHTIME"],
                on="HADM_ID",
                should_infer_icustay_id=False,
            )
        
        # sort by startdate
        spark_df = spark_df.sort(
            ["SUBJECT_ID", self.visit_key, "STARTDATE", "ENDDATE"],
            ascending=True
        )

        schema = StructType(
            [
                StructField("patient_id", StringType(), False),
                StructField("visit_id", StringType(), False),
                StructField("code", ArrayType(StringType()), False),
                StructField("name", ArrayType(StringType()), False),
                StructField("timestamp", ArrayType(StringType()), False),
                StructField("drug_type", ArrayType(StringType()), False),
                StructField("drug", ArrayType(StringType()), False),
                StructField("drug_name_poe", ArrayType(StringType()), False),
                StructField("drug_name_generic", ArrayType(StringType()), False),
                StructField("formulary_drug_cd", ArrayType(StringType()), False),
                StructField("gsn", ArrayType(StringType()), False),
                StructField("prod_strength", ArrayType(StringType()), False),
                StructField("dose_val_rx", ArrayType(StringType()), False),
                StructField("dose_unit_rx", ArrayType(StringType()), False),
                StructField("form_val_disp", ArrayType(StringType()), False),
                StructField("form_unit_disp", ArrayType(StringType()), False),
                StructField("route", ArrayType(StringType()), False)
            ]
        )
        @F.pandas_udf(schema, functionType=F.PandasUDFType.GROUPED_MAP)
        def prescription_unit(df):
            patient_id = str(df["SUBJECT_ID"].iloc[0])
            visit_id = str(df[self.visit_key].iloc[0])
            code = df["NDC"].tolist()
            name = ["NDC"] * len(code)
            timestamp = [str(t) for t in df["STARTDATE"].tolist()]
            drug_type = df["DRUG_TYPE"].tolist()
            drug = df["DRUG"].tolist()
            drug_name_poe = df["DRUG_NAME_POE"].tolist()
            drug_name_generic = df["DRUG_NAME_GENERIC"].tolist()
            formulary_drug_cd = df["FORMULARY_DRUG_CD"].tolist()
            gsn = df["GSN"].tolist()
            prod_strength = df["PROD_STRENGTH"].tolist()
            dose_val_rx = df["DOSE_VAL_RX"].tolist()
            dose_unit_rx = df["DOSE_UNIT_RX"].tolist()
            form_val_disp = df["FORM_VAL_DISP"].tolist()
            form_unit_disp = df["FORM_UNIT_DISP"].tolist()
            route = df["ROUTE"].tolist()
            return pd.DataFrame([[
                patient_id,
                visit_id,
                code,
                name,
                timestamp,
                drug_type,
                drug,
                drug_name_poe,
                drug_name_generic,
                formulary_drug_cd,
                gsn,
                prod_strength,
                dose_val_rx,
                dose_unit_rx,
                form_val_disp,
                form_unit_disp,
                route
            ]])

        pandas_df = spark_df.groupBy(
            "SUBJECT_ID", self.visit_key
        ).apply(prescription_unit).toPandas()

        def aggregate_events(row):
            events = []
            p_id = row["patient_id"]
            v_id = row["visit_id"]
            for (
                code, name, timestamp, drug_type, drug, drug_name_poe, drug_name_generic,
                formulary_drug_cd, gsn, prod_strength, dose_val_rx, dose_unit_rx, form_val_disp,
                form_unit_disp, route
            ) in zip(
                row["code"], row["name"], row["timestamp"], row["drug_type"], row["drug"],
                row["drug_name_poe"], row["drug_name_generic"], row["formulary_drug_cd"],
                row["gsn"], row["prod_strength"], row["dose_val_rx"], row["dose_unit_rx"],
                row["form_val_disp"], row["form_unit_disp"], row["route"]
            ):
                event = Event(
                    code=code,
                    name=name,
                    table=table,
                    vocabulary="NDC",
                    visit_id=v_id,
                    patient_id=p_id,
                    timestamp=strptime(timestamp),
                    drug_type=drug_type,
                    drug=drug,
                    drug_name_poe=drug_name_poe,
                    durg_name_generic=drug_name_generic,
                    formulary_drug_cd=formulary_drug_cd,
                    gsn=gsn,
                    prod_strength=prod_strength,
                    dose_val_rx=dose_val_rx,
                    dose_unit_rx=dose_unit_rx,
                    form_val_disp=form_val_disp,
                    form_unit_disp=form_unit_disp,
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
        spark: SparkSession,
    ) -> Dict[str, Patient]:
        """Helper function which parses LABEVENTS table.
        
        Will be called in `self.parse_tables()`.
        
        Docs:
            - LABEVENTS: https://mimic.mit.edu/docs/iii/tables/labevents/
        
        Args:
            patients: a dict of `Patient` objects indexed by patient_id.
            spark: a spark session for reading the table.
        
        Returns:
            The updated patients dict.
        """
        table = "LABEVENTS"
        # read tables
        spark_df = spark.read.csv(os.path.join(self.root, f"{table}.csv"), header=True)
        if self.dev:
            spark_df = spark_df.limit(3000)
        # convert dtype of charttime to datetime
        spark_df = spark_df.withColumn("CHARTTIME", F.to_timestamp("CHARTTIME"))

        if self.visit_unit == "icu":
            spark_df = self.filter_events(
                spark=spark,
                spark_df=spark_df,
                timestamp_key="CHARTTIME",
                joined_table="ICUSTAYS",
                select=["HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME"],
                on="HADM_ID",
                should_infer_icustay_id=True,
            )
        else:
            spark_df = self.filter_events(
                spark=spark,
                spark_df=spark_df,
                timestamp_key="CHARTTIME",
                joined_table="ADMISSIONS",
                select=["HADM_ID", "ADMITTIME", "DISCHTIME"],
                on="HADM_ID",
                should_infer_icustay_id=False,
            )
        # sort by charttime
        spark_df = spark_df.sort(["SUBJECT_ID", self.visit_key, "CHARTTIME"], ascending=True)

        schema = StructType(
            [
                StructField("patient_id", StringType(), False),
                StructField("visit_id", StringType(), False),
                StructField("code", ArrayType(StringType()), False),
                StructField("name", ArrayType(StringType()), False),
                StructField("timestamp", ArrayType(StringType()), False),
                StructField("value", ArrayType(StringType()), True),
                StructField("valuenum", ArrayType(StringType()), True),
                StructField("valueuom", ArrayType(StringType()), True),
                StructField("flag", ArrayType(StringType()), True)
            ]
        )
        @F.pandas_udf(schema, functionType=F.PandasUDFType.GROUPED_MAP)
        def lab_unit(df):
            patient_id = str(df["SUBJECT_ID"].iloc[0])
            visit_id = str(df[self.visit_key].iloc[0])
            code = df["ITEMID"].tolist()
            name = ["ITEMID"] * len(code)
            timestamp = [str(t) for t in df["CHARTTIME"].tolist()]
            value = df["VALUE"].tolist()
            valuenum = df["VALUENUM"].tolist()
            valueuom = df["VALUEUOM"].tolist()
            flag = df["FLAG"].tolist()
            return pd.DataFrame(
                [[patient_id, visit_id, code, name, timestamp, value, valuenum, valueuom, flag]]
            )

        pandas_df = spark_df.groupBy("SUBJECT_ID", self.visit_key).apply(lab_unit).toPandas()

        def aggregate_events(row):
            events = []
            p_id = row["patient_id"]
            v_id = row["visit_id"]
            for code, name, timestamp, value, valuenum, valueuom, flag in zip(
                row["code"], row["name"], row["timestamp"], row["value"], row["valuenum"],
                row["valueuom"], row["flag"]
            ):
                event = Event(
                    code=code,
                    name=name,
                    table=table,
                    vocabulary="MIMIC3_ITEMID",
                    visit_id=v_id,
                    patient_id=p_id,
                    timestamp=strptime(timestamp),
                    value=value,
                    valuenum=valuenum,
                    valueuom=valueuom,
                    flag=flag
                )
                events.append(event)
            return events

        # parallel apply to aggregate events
        aggregated_events = pandas_df.parallel_apply(aggregate_events, axis=1)

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, aggregated_events)
        return patients

    def parse_inputevents_mv(
        self,
        patients: Dict[str, Patient],
        spark: SparkSession
    ) -> Dict[str, Patient]:
        """Helper function which parses INPUTEVENTS_MV table.
        
        Will be called in `self.parse_tables()`.
        
        Docs:
            - INPUTEVENTS_MV: https://mimic.mit.edu/docs/iii/tables/inputevents_mv/
        
        Args:
            patients: a dict of `Patient` objects indexed by patient_id.
            spark: a spark session for reading the table.
        
        Returns:
            The updated patients dict.
        """
        table = "INPUTEVENTS_MV"

        # read tables
        spark_df = spark.read.csv(os.path.join(self.root, f"{table}.csv"), header=True)
        if self.dev:
            spark_df = spark_df.limit(3000)
        # convert dtype of starttime to datetime
        spark_df = spark_df.withColumn("STARTTIME", F.to_timestamp("STARTTIME"))
        
        if self.visit_unit == "icu":
            spark_df = self.filter_events(
                spark=spark,
                spark_df=spark_df,
                timestamp_key="STARTTIME",
                joined_table="ICUSTAYS",
                select=["ICUSTAY_ID", "INTIME", "OUTTIME"],
                on="ICUSTAY_ID",
                should_infer_icustay_id=False,
            )
        else:
            spark_df = self.filter_events(
                spark=spark,
                spark_df=spark_df,
                timestamp_key="STARTTIME",
                joined_table="ADMISSIONS",
                select=["HADM_ID", "ADMITTIME", "DISCHTIME"],
                on="HADM_ID",
                should_infer_icustay_id=False,
            )

        # sort by starttime
        spark_df = spark_df.sort(
            ["SUBJECT_ID", self.visit_key, "STARTTIME", "ENDTIME"],
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
                StructField("cancelreason", ArrayType(StringType()), False),
                StructField("comments_editedby", ArrayType(StringType()), False),
                StructField("comments_canceledby", ArrayType(StringType()), False),
                StructField("comments_date", ArrayType(StringType()), False),
                StructField("originalamount", ArrayType(StringType()), False),
                StructField("originalrate", ArrayType(StringType()), False)
            ]
        )
        @F.pandas_udf(schema, functionType=F.PandasUDFType.GROUPED_MAP)
        def inputevents_mv_unit(df):
            patient_id = str(df["SUBJECT_ID"].iloc[0])
            visit_id = str(df[self.visit_key].iloc[0])
            code = df["ITEMID"].tolist()
            name = ["ITEMID"] * len(code)
            timestamp = [str(t) for t in df["STARTTIME"].tolist()]
            amount = df["AMOUNT"].tolist()
            amountuom = df["AMOUNTUOM"].tolist()
            rate = df["RATE"].tolist()
            rateuom = df["RATEUOM"].tolist()
            orderid = df["ORDERID"].tolist()
            linkorderid = df["LINKORDERID"].tolist()
            ordercategoryname = df["ORDERCATEGORYNAME"].tolist()
            secondaryordercategoryname = df["SECONDARYORDERCATEGORYNAME"].tolist()
            ordercomponenttypedescription = df["ORDERCOMPONENTTYPEDESCRIPTION"].tolist()
            ordercategorydescription = df["ORDERCATEGORYDESCRIPTION"].tolist()
            patientweight = df["PATIENTWEIGHT"].tolist()
            totalamount = df["TOTALAMOUNT"].tolist()
            totalamountuom = df["TOTALAMOUNTUOM"].tolist()
            isopenbag = df["ISOPENBAG"].tolist()
            continueinnextdept = df["CONTINUEINNEXTDEPT"].tolist()
            cancelreason = df["CANCELREASON"].tolist()
            comments_editedby = df["COMMENTS_EDITEDBY"].tolist()
            comments_canceledby = df["COMMENTS_CANCELEDBY"].tolist()
            comments_date = df["COMMENTS_DATE"].tolist()
            originalamount = df["ORIGINALAMOUNT"].tolist()
            originalrate = df["ORIGINALRATE"].tolist()
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
                cancelreason,
                comments_editedby,
                comments_canceledby,
                comments_date,
                originalamount,
                originalrate
            ]])

        pandas_df = spark_df.groupBy(
            "SUBJECT_ID", self.visit_key
        ).apply(inputevents_mv_unit).toPandas()

        def aggregate_events(row):
            events = []
            p_id = row["patient_id"]
            v_id = row["visit_id"]
            for (
                code, name, timestamp, amount, amountuom, rate, rateuom, orderid, linkorderid, ordercategoryname,
                secondaryordercategoryname, ordercomponenttypedescription, ordercategorydescription,
                patientweight, totalamount, totalamountuom, isopenbag, continueinnexdept,
                cancelreason, comments_editedby, comments_canceledby, comments_date,
                originalamount, originalrate
            ) in zip(
                row["code"], row["name"], row["timestamp"], row["amount"], row["amountuom"],
                row["rate"], row["rateuom"], row["orderid"], row["linkorderid"],
                row["ordercategoryname"], row["secondaryordercategoryname"],
                row["ordercomponenttypedescription"], row["ordercategorydescription"],
                row["patientweight"], row["totalamount"], row["totalamountuom"], row["isopenbag"],
                row["continueinnextdept"], row["cancelreason"], row["comments_editedby"],
                row["comments_canceledby"], row["comments_date"], row["originalamount"],
                row["originalrate"]
            ):
                event = Event(
                    code=code,
                    name=name,
                    table=table,
                    vocabulary="MIMIC3_ITEMID",
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
                    continueinnexdept=continueinnexdept,
                    cancelreason=cancelreason,
                    comments_editedby=comments_editedby,
                    comments_canceledby=comments_canceledby,
                    comments_date=comments_date,
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

    def parse_inputevents_cv(
        self,
        patients: Dict[str, Patient],
        spark : SparkSession
    ) -> Dict[str, Patient]:
        """Helper function which parses INPUTEVENTS_CV table.
        
        Will be called in `self.parse_tables()`.
        
        Docs:
            - INPUTEVENTS_CV: https://mimic.mit.edu/docs/iii/tables/inputevents_cv/
        
        Args:
            patients: a dict of `Patient` objects indexed by patient_id.
            spark: a spark session for reading the table.
        
        Returns:
            The updated patients dict.
        """
        table = "INPUTEVENTS_CV"

        # read tables
        spark_df = spark.read.csv(os.path.join(self.root, f"{table}.csv"), header=True)
        if self.dev:
            spark_df = spark_df.limit(3000)
        # convert dtype of charttime to datetime
        spark_df = spark_df.withColumn("CHARTTIME", F.to_timestamp("CHARTTIME"))

        if self.visit_unit == "icu":
            spark_df = self.filter_events(
                spark=spark,
                spark_df=spark_df,
                timestamp_key="CHARTTIME",
                joined_table="ICUSTAYS",
                select=["ICUSTAY_ID", "INTIME", "OUTTIME"],
                on="ICUSTAY_ID",
                should_infer_icustay_id=False,
            )
        else:
            spark_df = self.filter_events(
                spark=spark,
                spark_df=spark_df,
                timestamp_key="CHARTTIME",
                joined_table="ADMISSIONS",
                select=["HADM_ID", "ADMITTIME", "DISCHTIME"],
                on="HADM_ID",
                should_infer_icustay_id=False,
            )
        
        # sort by charttime
        spark_df = spark_df.sort(["SUBJECT_ID", self.visit_key, "CHARTTIME"], ascending=True)
        
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
                StructField("cgid", ArrayType(StringType()), False),
                StructField("orderid", ArrayType(StringType()), False),
                StructField("linkorderid", ArrayType(StringType()), False),
                StructField("stopped", ArrayType(StringType()), False),
                StructField("newbottle", ArrayType(StringType()), False),
                StructField("originalamount", ArrayType(StringType()), False),
                StructField("originalamountuom", ArrayType(StringType()), False),
                StructField("originalroute", ArrayType(StringType()), False),
                StructField("originalrate", ArrayType(StringType()), False),
                StructField("originalrateuom", ArrayType(StringType()), False),
                StructField("originalsite", ArrayType(StringType()), False)
            ]
        )
        @F.pandas_udf(schema, functionType=F.PandasUDFType.GROUPED_MAP)
        def inputevents_cv_unit(df):
            patient_id = str(df["SUBJECT_ID"].iloc[0])
            visit_id = str(df[self.visit_key].iloc[0])
            code = df["ITEMID"].tolist()
            name = ["ITEMID"] * len(code)
            timestamp = [str(t) for t in df["CHARTTIME"].tolist()]
            amount = df["AMOUNT"].tolist()
            amountuom = df["AMOUNTUOM"].tolist()
            rate = df["RATE"].tolist()
            rateuom = df["RATEUOM"].tolist()
            cgid = df["CGID"].tolist()
            orderid = df["ORDERID"].tolist()
            linkorderid = df["LINKORDERID"].tolist()
            stopped = df["STOPPED"].tolist()
            newbottle = df["NEWBOTTLE"].tolist()
            originalamount = df["ORIGINALAMOUNT"].tolist()
            originalamountuom = df["ORIGINALAMOUNTUOM"].tolist()
            originalroute = df["ORIGINALROUTE"].tolist()
            originalrate = df["ORIGINALRATE"].tolist()
            originalrateuom = df["ORIGINALRATEUOM"].tolist()
            originalsite = df["ORIGINALSITE"].tolist()
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
                cgid,
                orderid,
                linkorderid,
                stopped,
                newbottle,
                originalamount,
                originalamountuom,
                originalroute,
                originalrate,
                originalrateuom,
                originalsite
            ]])
        
        pandas_df = spark_df.groupBy(
            "SUBJECT_ID", self.visit_key
        ).apply(inputevents_cv_unit).toPandas()
        
        def aggregate_events(row):
            events = []
            p_id = row["patient_id"]
            v_id = row["visit_id"]
            for (
                code, name, timestamp, amount, amountuom, rate, rateuom, cgid, orderid, linkorderid,
                stopped, newbottle, originalamount, originalamountuom, originalroute, originalrate,
                originalrateuom, originalsite
            ) in zip(
                row["code"], code["name"], row["timestamp"], row["amount"], row["amountuom"],
                row["rate"], row["rateuom"], row["cgid"], row["orderid"], row["linkorderid"],
                row["stopped"], row["newbottle"], row["originalamount"], row["originalamountuom"],
                row["originalroute"], row["originalrate"], row["originalrateuom"],
                row["originalsite"]
            ):
                event = Event(
                    code=code,
                    name=name,
                    table=table,
                    vocabulary="MIMIC3_ITEMID",
                    visit_id=v_id,
                    patient_id=p_id,
                    timestamp=strptime(timestamp),
                    amount=amount,
                    amountuom=amountuom,
                    rate=rate,
                    rateuom=rateuom,
                    cgid=cgid,
                    orderid=orderid,
                    linkorderid=linkorderid,
                    stopped=stopped,
                    newbottle=newbottle,
                    originalamount=originalamount,
                    originalamountuom=originalamountuom,
                    originalroute=originalroute,
                    originalrate=originalrate,
                    originalrateuom=originalrateuom,
                    originalsite=originalsite
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
        """Helper function which parses CHARTEVENTS table.
        
        Will be called in `self.parse_tables()`.
        
        Docs:
            - CHARTEVENTS: https://mimic.mit.edu/docs/iii/tables/chartevents/
        
        Args:
            patients: a dict of `Patient` objects indexed by patient_id.
            spark: a spark session for reading the table.
        
        Returns:
            The updated patients dict.
        """
        table = "CHARTEVENTS"

        # read tables
        spark_df = spark.read.csv(os.path.join(self.root, f"{table}.csv"), header=True)
        if self.dev:
            spark_df = spark_df.limit(3000)
        # convert dtype of charttime to datetime
        spark_df = spark_df.withColumn("CHARTTIME", F.to_timestamp("CHARTTIME"))

        if self.visit_unit == "icu":
            spark_df = self.filter_events(
                spark=spark,
                spark_df=spark_df,
                timestamp_key="CHARTTIME",
                joined_table="ICUSTAYS",
                select=["ICUSTAY_ID", "INTIME", "OUTTIME"],
                on="ICUSTAY_ID",
                should_infer_icustay_id=False
            )
        else:
            spark_df = self.filter_events(
                spark=spark,
                spark_df=spark_df,
                timestamp_key="CHARTTIME",
                joined_table="ADMISSIONS",
                select=["HADM_ID", "ADMITTIME", "DISCHTIME"],
                on="HADM_ID",
                should_infer_icustay_id=False,
            )

        # sort by charttime
        spark_df = spark_df.sort(["SUBJECT_ID", self.visit_key, "CHARTTIME"], ascending=True)

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
                StructField("error", ArrayType(StringType()), False),
                StructField("resultstatus", ArrayType(StringType()), False),
                StructField("stopped", ArrayType(StringType()), False),
            ]
        )
        @F.pandas_udf(schema, functionType=F.PandasUDFType.GROUPED_MAP)
        def chartevents_unit(df):
            patient_id = str(df["SUBJECT_ID"].iloc[0])
            visit_id = str(df[self.visit_key].iloc[0])
            code = df["ITEMID"].tolist()
            name = ["ITEMID"] * len(code)
            timestamp = [str(t) for t in df["CHARTTIME"].tolist()]
            value = df["VALUE"].tolist()
            valuenum = df["VALUENUM"].tolist()
            valueuom = df["VALUEUOM"].tolist()
            warning = df["WARNING"].tolist()
            error = df["ERROR"].tolist()
            resultstatus = df["RESULTSTATUS"].tolist()
            stopped = df["STOPPED"].tolist()
            return pd.DataFrame([[
                patient_id,
                visit_id,
                code,
                name,
                timestamp,
                value,
                valuenum,
                valueuom,
                warning,
                error,
                resultstatus,
                stopped
            ]])

        pandas_df = spark_df.groupBy(
            "SUBJECT_ID", self.visit_key
        ).apply(chartevents_unit).toPandas()

        def aggregate_events(row):
            events = []
            p_id = row["patient_id"]
            v_id = row["visit_id"]
            for (
                code, name, timestamp, value, valuenum, valueuom, warning, error, resultstatus, stopped
            ) in zip(
                row["code"], row["name"], row["timestamp"], row["value"], row["valuenum"],
                row["valueuom"], row["warning"], row["error"], row["resultstatus"], row["stopped"]
            ):
                event = Event(
                    code=code,
                    name=name,
                    table=table,
                    vocabulary="MIMIC3_ITEMID",
                    visit_id=v_id,
                    patient_id=p_id,
                    timestamp=strptime(timestamp),
                    value=value,
                    valuenum=valuenum,
                    valueuom=valueuom,
                    warning=warning,
                    error=error,
                    resultstatus=resultstatus,
                    stopped=stopped
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
        """Helper function which parses OUTPUTEVENTS table.
        
        Will be called in `self.parse_tables()`.
        
        Docs:
            - OUTPUTEVENTS: https://mimic.mit.edu/docs/iii/tables/outputevents/
        
        Args:
            patients: a dict of `Patient` objects indexed by patient_id.
            spark: a spark session for reading the table.
        
        Returns:
            The updated patients dict.
        """
        table = "OUTPUTEVENTS"

        # read tables
        spark_df = spark.read.csv(os.path.join(self.root, f"{table}.csv"), header=True)
        if self.dev:
            spark_df = spark_df.limit(3000)
        # convert dtype of charttime to datetime
        spark_df = spark_df.withColumn("CHARTTIME", F.to_timestamp("CHARTTIME"))

        if self.visit_unit == "icu":
            spark_df = self.filter_events(
                spark=spark,
                spark_df=spark_df,
                timestamp_key="CHARTTIME",
                joined_table="ICUSTAYS",
                select=["ICUSTAY_ID", "INTIME", "OUTTIME"],
                on="ICUSTAY_ID",
                should_infer_icustay_id=False
            )
        else:
            spark_df = self.filter_events(
                spark=spark,
                spark_df=spark_df,
                timestamp_key="CHARTTIME",
                joined_table="ADMISSIONS",
                select=["HADM_ID", "ADMITTIME", "DISCHTIME"],
                on="HADM_ID",
                should_infer_icustay_id=False,
            )

        # sort by charttime
        spark_df = spark_df.sort(["SUBJECT_ID", self.visit_key, "CHARTTIME"], ascending=True)

        schema = StructType(
            [
                StructField("patient_id", StringType(), False),
                StructField("visit_id", StringType(), False),
                StructField("code", ArrayType(StringType()), False),
                StructField("name", ArrayType(StringType()), False),
                StructField("timestamp", ArrayType(StringType()), False),
                StructField("value", ArrayType(StringType()), False),
                StructField("valueuom", ArrayType(StringType()), False),
                StructField("stopped", ArrayType(StringType()), False),
                StructField("newbottle", ArrayType(StringType()), False),
                StructField("iserror", ArrayType(StringType()), False),
            ]
        )
        @F.pandas_udf(schema, functionType=F.PandasUDFType.GROUPED_MAP)
        def outputevents_unit(df):
            patient_id = str(df["SUBJECT_ID"].iloc[0])
            visit_id = str(df[self.visit_key].iloc[0])
            code = df["ITEMID"].tolist()
            name = ["ITEMID"] * len(code)
            timestamp = [str(t) for t in df["CHARTTIME"].tolist()]
            value = df["VALUE"].tolist()
            valueuom = df["VALUEUOM"].tolist()
            stopped = df["STOPPED"].tolist()
            newbottle = df["NEWBOTTLE"].tolist()
            iserror = df["ISERROR"].tolist()
            return pd.DataFrame([[
                patient_id,
                visit_id,
                code,
                name,
                timestamp,
                value,
                valueuom,
                stopped,
                newbottle,
                iserror
            ]])

        pandas_df = spark_df.groupBy(
            "SUBJECT_ID", self.visit_key
        ).apply(outputevents_unit).toPandas()

        def aggregate_events(row):
            events = []
            p_id = row["patient_id"]
            v_id = row["visit_id"]
            for (
                code, name, timestamp, value, valueuom, stopped, newbottle, iserror
            ) in zip(
                row["code"], row["name"], row["timestamp"], row["value"], row["valueuom"],
                row["stopped"], row["newbottle"], row["iserror"]
            ):
                event = Event(
                    code=code,
                    name=name,
                    table=table,
                    vocabulary="MIMIC3_ITEMID",
                    visit_id=v_id,
                    patient_id=p_id,
                    timestamp=strptime(timestamp),
                    value=value,
                    valueuom=valueuom,
                    stopped=stopped,
                    newbottle=newbottle,
                    iserror=iserror
                )
                events.append(event)
            return events
        
        # parallel apply to aggregate events
        aggregated_events = pandas_df.parallel_apply(aggregate_events, axis=1)

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, aggregated_events)
        return patients

    def parse_procedureevents_mv(
        self,
        patients: Dict[str, Patient],
        spark: SparkSession,
    ) -> Dict[str, Patient]:
        """Helper function which parses PROCEDUREEVENTS table.
        
        Will be called in `self.parse_tables()`.
        
        Docs:
            - PROCEDUREEVENTS: https://mimic.mit.edu/docs/iii/tables/procedureevents_mv/
        
        Args:
            patients: a dict of `Patient` objects indexed by patient_id.
            spark: a spark session for reading the table.
        
        Returns:
            The updated patients dict.
        """
        table = "PROCEDUREEVENTS_MV"

        # read tables
        spark_df = spark.read.csv(os.path.join(self.root, f"{table}.csv"), header=True)
        if self.dev:
            spark_df = spark_df.limit(3000)
        # convert dtype of charttime to datetime
        spark_df = spark_df.withColumn("STARTTIME", F.to_timestamp("STARTTIME"))

        if self.visit_unit == "icu":
            spark_df = self.filter_events(
                spark=spark,
                spark_df=spark_df,
                timestamp_key="STARTTIME",
                joined_table="ICUSTAYS",
                select=["ICUSTAY_ID", "INTIME", "OUTTIME"],
                on="ICUSTAY_ID",
                should_infer_icustay_id=False
            )
        else:
            spark_df = self.filter_events(
                spark=spark,
                spark_df=spark_df,
                timestamp_key="STARTTIME",
                joined_table="ADMISSIONS",
                select=["HADM_ID", "ADMITTIME", "DISCHTIME"],
                on="HADM_ID",
                should_infer_icustay_id=False,
            )

        # sort by charttime
        spark_df = spark_df.sort(["SUBJECT_ID", self.visit_key, "STARTTIME", "ENDTIME"], ascending=True)

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
                StructField("secondaryordercategoryname", ArrayType(StringType()), False),
                StructField("ordercategorydescription", ArrayType(StringType()), False),
                StructField("isopenbag", ArrayType(StringType()), False),
                StructField("continueinnextdept", ArrayType(StringType()), False),
                StructField("cancelreason", ArrayType(StringType()), False),
                StructField("statusdescription", ArrayType(StringType()), False),
                StructField("comments_editedby", ArrayType(StringType()), False),
                StructField("comments_canceledby", ArrayType(StringType()), False),
                StructField("comments_date", ArrayType(StringType()), False),
            ]
        )
        @F.pandas_udf(schema, functionType=F.PandasUDFType.GROUPED_MAP)
        def procedureevents_mv_unit(df):
            patient_id = str(df["SUBJECT_ID"].iloc[0])
            visit_id = str(df[self.visit_key].iloc[0])
            code = df["ITEMID"].tolist()
            name = ["ITEMID"] * len(code)
            timestamp = [str(t) for t in df["STARTTIME"].tolist()]
            value = df["VALUE"].tolist()
            valueuom = df["VALUEUOM"].tolist()
            location = df["LOCATION"].tolist()
            locationcategory = df["LOCATIONCATEGORY"].tolist()
            orderid = df["ORDERID"].tolist()
            linkorderid = df["LINKORDERID"].tolist()
            ordercategoryname = df["ORDERCATEGORYNAME"].tolist()
            secondaryordercategoryname = df["SECONDARYORDERCATEGORYNAME"].tolist()
            ordercategorydescription = df["ORDERCATEGORYDESCRIPTION"].tolist()
            isopenbag = df["ISOPENBAG"].tolist()
            continueinnextdept = df["CONTINUEINNEXTDEPT"].tolist()
            cancelreason = df["CANCELREASON"].tolist()
            statusdescription = df["STATUSDESCRIPTION"].tolist()
            comments_editedby = df["COMMENTS_EDITEDBY"].tolist()
            comments_canceledby = df["COMMENTS_CANCELEDBY"].tolist()
            comments_date = df["COMMENTS_DATE"].tolist()
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
                secondaryordercategoryname,
                ordercategorydescription,
                isopenbag,
                continueinnextdept,
                cancelreason,
                statusdescription,
                comments_editedby,
                comments_canceledby,
                comments_date
            ]])

        pandas_df = spark_df.groupBy(
            "SUBJECT_ID", self.visit_key
        ).apply(procedureevents_mv_unit).toPandas()

        def aggregate_events(row):
            events = []
            p_id = row["patient_id"]
            v_id = row["visit_id"]
            for (
                code, name, timestamp, value, valueuom, location, locationcategory, orderid,
                linkorderid, ordercategoryname, secondaryordercategoryname,
                ordercategorydescription, isopenbag, continueinnextdept, cancelreason,
                statusdescription, comments_editedby, comments_canceledby, comments_date
            ) in zip(
                row["code"], row["name"], row["timestamp"], row["value"],row["vluaeuom"],
                row["location"], row["locationcategory"], row["orderid"], row["linkorderid"],
                row["ordercategoryname"], row["secondaryordercategoryname"],
                row["ordercategorydescription"], row["isopenbag"], row["continueinnextdept"],
                row["cancelreason"], row["statusdescription"], row["comments_editedby"],
                row["comments_canceledby"], row["comments_date"]
            ):
                event = Event(
                    code=code,
                    name=name,
                    table=table,
                    vocabulary="MIMIC3_ITEMID",
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
                    secondaryordercategoryname=secondaryordercategoryname,
                    ordercategorydescription=ordercategorydescription,
                    isopenbag=isopenbag,
                    continueinnextdept=continueinnextdept,
                    cancelreason=cancelreason,
                    statusdescription=statusdescription,
                    comments_editedby=comments_editedby,
                    comments_canceledby=comments_canceledby,
                    commens_date=comments_date
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
        """Helper function which parses MICROBIOLOGYEVENTS table.
        
        Will be called in `self.parse_tables()`.
        
        Docs:
            - MICROBIOLOGYEVENTS: https://mimic.mit.edu/docs/iii/tables/microbiologyevents/
        
        Args:
            patients: a dict of `Patient` objects indexed by patient_id.
            spark: a spark session for reading the table.
        
        Returns:
            The updated patients dict.
        """
        table = "MICROBIOLOGYEVENTS"

        # read tables
        spark_df = spark.read.csv(os.path.join(self.root, f"{table}.csv"), header=True)
        if self.dev:
            spark_df = spark_df.limit(3000)
        # convert dtype of charttime to datetime
        spark_df = spark_df.withColumn("CHARTTIME", F.to_timestamp("CHARTTIME"))

        if self.visit_unit == "icu":
            spark_df = self.filter_events(
                spark=spark,
                spark_df=spark_df,
                timestamp_key="CHARTTIME",
                joined_table="ICUSTAYS",
                select=["HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME"],
                on="HADM_ID",
                should_infer_icustay_id=True,
            )
        else:
            spark_df = self.filter_events(
                spark=spark,
                spark_df=spark_df,
                timestamp_key="CHARTTIME",
                joined_table="ADMISSIONS",
                select=["HADM_ID", "ADMITTIME", "DISCHTIME"],
                on="HADM_ID",
                should_infer_icustay_id=False,
            )

        # sort by charttime
        spark_df = spark_df.sort(["SUBJECT_ID", self.visit_key, "CHARTTIME"], ascending=True)

        schema = StructType(
            [
                StructField("patient_id", StringType(), False),
                StructField("visit_id", StringType(), False),
                StructField("code", ArrayType(StringType()), False),
                StructField("name", ArrayType(StringType()), False),
                StructField("timestamp", ArrayType(StringType()), False),
                StructField("spec_itemid", ArrayType(StringType()), False),
                StructField("spec_type_desc", ArrayType(StringType()), False),
                StructField("org_itemid", ArrayType(StringType()), False),
                StructField("org_name", ArrayType(StringType()), False),
                StructField("isolate_num", ArrayType(StringType()), False),
                StructField("ab_itemid", ArrayType(StringType()), False),
                StructField("ab_name", ArrayType(StringType()), False),
                StructField("dilution_text", ArrayType(StringType()), False),
                StructField("dilution_comparison", ArrayType(StringType()), False),
                StructField("dilution_value", ArrayType(StringType()), False),
                StructField("interpretation", ArrayType(StringType()), False),
            ]
        )
        @F.pandas_udf(schema, functionType=F.PandasUDFType.GROUPED_MAP)
        def microbiologyevents_unit(df):
            patient_id = str(df["SUBJECT_ID"].iloc[0])
            visit_id = str(df[self.visit_key].iloc[0])
            code = [None] * len(df)
            name = [None] * len(code)
            timestamp = [str(t) for t in df["CHARTTIME"].tolist()]
            spec_itemid = df["SPEC_ITEMID"].tolist()
            spec_type_desc = df["SPEC_TYPE_DESC"].tolist()
            org_itemid = df["ORG_ITEMID"].tolist()
            org_name = df["ORG_NAME"].tolist()
            isolate_num = df["ISOLATE_NUM"].tolist()
            ab_itemid = df["AB_ITEMID"].tolist()
            ab_name = df["AB_NAME"].tolist()
            dilution_text = df["DILUTION_TEXT"].tolist()
            dilution_comparison = df["DILUTION_COMPARISON"].tolist()
            dilution_value = df["DILUTION_VALUE"].tolist()
            interpretation = df["INTERPRETATION"].tolist()
            return pd.DataFrame([[
                patient_id,
                visit_id,
                code,
                name,
                timestamp,
                spec_itemid,
                spec_type_desc,
                org_itemid,
                org_name,
                isolate_num,
                ab_itemid,
                ab_name,
                dilution_text,
                dilution_comparison,
                dilution_value,
                interpretation
            ]])

        pandas_df = spark_df.groupBy(
            "SUBJECT_ID", self.visit_key
        ).apply(microbiologyevents_unit).toPandas()

        def aggregate_events(row):
            events = []
            p_id = row["patient_id"]
            v_id = row["visit_id"]
            for (
                code, name, timestamp, spec_itemid, spec_type_desc, org_itemid, org_name,
                isolate_num, ab_itemid, ab_name, dilution_text, dilution_comparison,
                diluition_value, interpretation
            ) in zip(
                row["code"], row["name"], row["timestamp"], row["spec_itemid"],
                row["spec_type_desc"], row["org_itemid"], row["org_name"], row["isolate_num"],
                row["ab_itemid"], row["ab_name"], row["dilution_text"], row["dilution_comparison"],
                row["dilution_value"], row["interpretation"]
            ):
                event = Event(
                    code=code,
                    name=name,
                    table=table,
                    vocabulary="MIMIC3_ITEMID",
                    visit_id=v_id,
                    patient_id=p_id,
                    timestamp=strptime(timestamp),
                    spec_itemid=spec_itemid,
                    spec_type_desc=spec_type_desc,
                    org_itemid=org_itemid,
                    org_name=org_name,
                    isolate_num=isolate_num,
                    ab_itemid=ab_itemid,
                    ab_name=ab_name,
                    dilution_text=dilution_text,
                    dilution_comparison=dilution_comparison,
                    diluition_value=diluition_value,
                    interpretation=interpretation
                )
                events.append(event)
            return events

        # parallel apply to aggregate events
        aggregated_events = pandas_df.parallel_apply(aggregate_events, axis=1)

        # summarize the results
        patients = self._add_events_to_patient_dict(patients, aggregated_events)
        return patients

if __name__ == "__main__":
    dataset = MIMIC3Dataset(
        root="https://storage.googleapis.com/pyhealth/mimiciii-demo/1.4/",
        tables=[
            "DIAGNOSES_ICD",
            "PROCEDURES_ICD",
            "PRESCRIPTIONS",
            "LABEVENTS",
        ],
        code_mapping={"NDC": "ATC"},
        dev=True,
        refresh_cache=True,
    )
    dataset.stat()
    dataset.info()

    # dataset = MIMIC3Dataset(
    #     root="/srv/local/data/physionet.org/files/mimiciii/1.4",
    #     tables=["DIAGNOSES_ICD", "PRESCRIPTIONS"],
    #     dev=True,
    #     code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
    #     refresh_cache=False,
    # )
    # print(dataset.stat())
    # print(dataset.available_tables)
    # print(list(dataset.patients.values())[4])
