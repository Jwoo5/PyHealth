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

class MIMIC4SparkDataset(BaseEHRSparkDataset):
    """TODO: to be written
    The MIMIC-IV dataset is a large dataset of de-identified health records of ICU
    patients. The dataset is available at https://mimic.physionet.org/.

    The basic information is stored in the following tables:
        - patients: defines a patient in the database, subject_id.
        - admission: define a patient's hospital admission, hadm_id.
        - icustays: defines a patient's ICU stay, stay_id.    
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
        # read icustays table
        icustays_df = pd.read_csv(
            os.path.join(self.root, "icustays.csv"),
            dtype={"subject_id": str, "stay_id": str}
        )

        # merge patient, admission, and icustay tables
        df = pd.merge(
            patients_df,
            admissions_df[[
                "subject_id",
                "hadm_id",
                "admittime",
                "dischtime",
                "race",
                "hospital_expire_flag"
            ]],
            on="subject_id",
            how="inner"
        )
        df = pd.merge(
            df,
            icustays_df[["subject_id", "stay_id", "intime", "outtime"]],
            on="subject_id",
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
        """TODO: to be written"""
        #TODO
        raise NotImplementedError()

    def parse_procedures_icd(self, patients: Dict[str, Patient]) -> Dict[str, Patient]:
        """TODO: to be written"""
        #TODO
        raise NotImplementedError()

    def parse_prescriptions(
        self,
        patients: Dict[str, Patient],
        spark: SparkSession
    ) -> Dict[str, Patient]:
        """TODO: to be written"""
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
                code, timestamp, pharmacy_id, poe_id, poe_seq, drug_type, drug, formulary_drug_cd,
                gsn, prod_strength, form_rx, dose_val_rx, dose_unit_rx, form_val_disp,
                form_unit_disp, doses_per_24_hrs, route
            ) in zip(
                row["code"], row["timestamp"], row["pharmacy_id"], row["poe_id"], row["poe_seq"],
                row["drug_type"], row["drug"], row["formulary_drug_cd"], row["gsn"],
                row["prod_strength"], row["form_rx"], row["dose_val_rx"], row["dose_unit_rx"],
                row["form_val_disp"], row["form_unit_disp"], row["doses_per_24_hrs"], row["route"]
            ):
                event = Event(
                    code=code,
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
        """TODO: to be written"""
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
                code, timestamp, value, valuenum, valueuom, ref_range_lower, ref_range_upper,
                flag, priority, comments
            ) in zip(
                row["code"], row["timestamp"], row["value"], row["valuenum"], row["valueuom"],
                row["ref_range_lower"], row["ref_range_upper"], row["flag"], row["priority"],
                row["comments"]
            ):
                event = Event(
                    code=code,
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
        """TODO: to be written"""
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
                code, timestamp, amount, amountuom, rate, rateuom, orderid, linkorderid,
                ordercategoryname, secondaryordercategoryname, ordercomponenttypedescription,
                ordercategorydescription, patientweight, totalamount, totalamountuom, isopenbag,
                continueinnextdept, statusdescription, originalamount, originalrate
            ) in zip(
                row["code"], row["timestamp"], row["amount"], row["amountuom"], row["rate"],
                row["rateuom"], row["orderid"], row["linkorderid"], row["ordercategoryname"],
                row["secondaryordercategoryname"], row["ordercomponenttypedescription"],
                row["ordercategorydescription"], row["patientweight"], row["totalamount"],
                row["totalamountuom"], row["isopenbag"], row["continueinnextdept"],
                row["statusdescription"], row["originalamount"], row["originalrate"]
            ):
                event = Event(
                    code=code,
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

if __name__ == "__main__":
    dataset = MIMIC4Dataset(
        root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",
        tables=["diagnoses_icd", "procedures_icd", "prescriptions", "labevents"],
        code_mapping={"NDC": "ATC"},
        refresh_cache=False,
    )
    dataset.stat()
    dataset.info()
