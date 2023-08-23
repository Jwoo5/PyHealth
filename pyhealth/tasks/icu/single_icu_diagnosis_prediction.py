import os
import pickle
from typing import List, Dict
import treelib
from collections import Counter

import pandas as pd
import numpy as np

from pyhealth.data import Patient
from pyhealth.tasks import register_task
from pyhealth.datasets.utils import hash_str
from pyhealth import BASE_CACHE_PATH

DATA_CACHE_PATH = os.path.join(BASE_CACHE_PATH, "data")

def icd10_to_icd9(icd9_or_10_codes, icd_versions):
    if not hasattr(icd10_to_icd9, "gem"):
        filename = hash_str("icd10cmtoicd9gem") + ".pkl"
        if not os.path.exists(os.path.join(DATA_CACHE_PATH, filename)):
            import subprocess
            subprocess.run(
                [
                    "wget", "-N", "-c",
                    "https://data.nber.org/gem/icd10cmtoicd9gem.csv",
                    "-P", DATA_CACHE_PATH
                ]
            )
            os.rename(
                os.path.join(DATA_CACHE_PATH, "icd10cmtoicd9gem.csv"),
                os.path.join(DATA_CACHE_PATH, filename)
            )
            
            gem = pd.read_csv(os.path.join(DATA_CACHE_PATH, filename))
            with open(os.path.join(DATA_CACHE_PATH, filename), "wb") as f:
                pickle.dump(gem, f)
    
        with open(os.path.join(DATA_CACHE_PATH, filename), "rb") as f:
            gem = pickle.load(f)
            icd10_to_icd9.map_cms = dict(zip(gem["icd10cm"], gem["icd9cm"]))
            icd10_to_icd9.gem = gem
            icd10_to_icd9.map_manual = dict()

    assert len(icd9_or_10_codes) == len(icd_versions)
    
    result = set()
    for icd_code, icd_version in zip(icd9_or_10_codes, icd_versions):
        if icd_version == 9:
            result.add(icd_code)
        elif icd_version == 10:
            if icd_code in icd10_to_icd9.map_cms:
                result.add(icd10_to_icd9.map_cms[icd_code])
            else:
                if icd_code in icd10_to_icd9.map_manual:
                    result.add(icd10_to_icd9.map_manual[icd_code])
                else:
                    for i in range(len(icd_code), 0, -1):
                        subcode_10 = icd_code[:i]
                        if subcode_10 in icd10_to_icd9.gem["icd10cm"]:
                            icd_code_9 = (
                                icd10_to_icd9.gem[
                                    icd10_to_icd9.gem["icd10cm"].str.contains(subcode_10)
                                ]["icd9cm"].mode().iloc[0]
                            )
                            icd10_to_icd9.map_manual[icd_code] = icd_code_9
                            result.add(icd_code_9)
                            break
                    if icd_code not in icd10_to_icd9.map_manual:
                        result.add("N/A")
        else:
            raise AssertionError(
                f"Invalid icd version. {icd_code}: {icd_version}"
            )
    
    return list(result)

def categorize_diagnosis_codes(icd9_codes: List[str]):
    if not hasattr(categorize_diagnosis_codes, "ccs_map"):
        filename = hash_str("ccs_map") + ".pkl"
        if not os.path.exists(os.path.join(DATA_CACHE_PATH, filename)):
            import subprocess
            import zipfile
            import shutil
            subprocess.run(
                [
                    "wget", "-N", "-c",
                    "https://www.hcup-us.ahrq.gov/toolssoftware/ccs/Multi_Level_CCS_2015.zip",
                    "-P", DATA_CACHE_PATH
                ]
            )
            with zipfile.ZipFile(
                os.path.join(DATA_CACHE_PATH, "Multi_Level_CCS_2015.zip"), "r"
            ) as zip_ref:
                zip_ref.extractall(os.path.join(DATA_CACHE_PATH, hash_str("foo.d")))
            os.rename(
                os.path.join(DATA_CACHE_PATH, hash_str("foo.d"), "ccs_multi_dx_tool_2015.csv"),
                os.path.join(DATA_CACHE_PATH, filename)
            )
            os.remove(os.path.join(DATA_CACHE_PATH, "Multi_Level_CCS_2015.zip"))
            shutil.rmtree(os.path.join(DATA_CACHE_PATH, hash_str("foo.d")))

            ccs_df = pd.read_csv(os.path.join(DATA_CACHE_PATH, filename))
            ccs_df["'ICD-9-CM CODE'"] = ccs_df["'ICD-9-CM CODE'"].str[1:-1].str.strip()
            ccs_df["'CCS LVL 1'"] = ccs_df["'CCS LVL 1'"].str[1:-1]
            ccs_map = {
                x: int(y) - 1 for _, (x, y) in ccs_df[["'ICD-9-CM CODE'", "'CCS LVL 1'"]].iterrows()
            }
            with open(os.path.join(DATA_CACHE_PATH, filename), "wb") as f:
                pickle.dump(ccs_map, f)

        with open(os.path.join(DATA_CACHE_PATH, filename), "rb") as f:
            categorize_diagnosis_codes.ccs_map = pickle.load(f)
            categorize_diagnosis_codes.num_classes = len(
                set(categorize_diagnosis_codes.ccs_map.values())
            )

    output = [
        categorize_diagnosis_codes.ccs_map[x] for x in icd9_codes
        if x in categorize_diagnosis_codes.ccs_map
    ]

    # we don't use category number 14 since it is very rare case
    # accordingly, category number >= 15 should be shifted by 1
    output = sorted(list(set([x if x < 14 else x - 1 for x in output if x != 14])))
    multi_hot_vector = np.zeros(categorize_diagnosis_codes.num_classes - 1, dtype=int)
    multi_hot_vector[output] = 1
    multi_hot_vector = multi_hot_vector.tolist()

    return multi_hot_vector

@register_task("single_icu_diagnosis_prediction_mimic3")
def single_icu_diagnosis_prediction_mimic3_fn(
    patient: Patient,
    **kwargs
) -> Dict[str, int]:
    """TODO: to be written"""
    labels = {}

    for visit in patient:
        diagnosis_categories = categorize_diagnosis_codes(visit.diagnosis_codes)

        labels[visit.visit_id] = diagnosis_categories

    return labels

@register_task("single_icu_diagnosis_prediction_mimic4")
def single_icu_diagnosis_prediction_mimic4_fn(
    patient: Patient,
    **kwargs
) -> Dict[str, int]:
    """TODO: to be written"""
    labels = {}
    
    for visit in patient:
        icd9_codes = icd10_to_icd9(visit.diagnosis_codes, visit.icd_versions)
        diagnosis_categories = categorize_diagnosis_codes(icd9_codes)

        labels[visit.visit_id] = diagnosis_categories

    return labels

@register_task("single_icu_diagnosis_prediction_eicu")
def single_icu_diagnosis_prediction_eicu_fn(
    patient: Patient,
    root: str = None,
    **kwargs
) -> Dict[str, int]:
    """TODO: to be written"""
    labels = {}
    
    if not hasattr(categorize_diagnosis_codes, "ccs_map"):
        categorize_diagnosis_codes([])

    for visit in patient:
        assert len(visit.diagnosis_string) == len(visit.diagnosis_codes)
        
        dx_codes = set()
        dx_cats_for_unmatched = []
        for dx_str, dx_code in zip(visit.diagnosis_string, visit.diagnosis_codes):
            if str(dx_code) == "nan":
                continue

            codes = [x.strip().replace(".", "") for x in dx_code.split(",")]
            for code in codes:
                if code not in categorize_diagnosis_codes.ccs_map:
                    code = icd10_to_icd9([code], [10])[0]

                if code == "N/A":
                    if not hasattr(single_icu_diagnosis_prediction_eicu_fn, "dx_code_tree"):
                        filename = hash_str("str_to_dx_cat") + ".pkl"
                        if not os.path.exists(os.path.join(DATA_CACHE_PATH, filename)):
                            diagnoses_df = pd.read_csv(
                                os.path.join(root, "diagnosis.csv")
                            )[["diagnosisstring", "icd9code"]]
                            str_to_dx_cat = diagnoses_df.dropna(subset=["icd9code"])
                            str_to_dx_cat = str_to_dx_cat.groupby("diagnosisstring").first().reset_index()
                            str_to_dx_cat["icd9code"] = str_to_dx_cat["icd9code"].str.split(",")
                            str_to_dx_cat = str_to_dx_cat.explode("icd9code")
                            str_to_dx_cat["icd9code"] = str_to_dx_cat["icd9code"].str.strip()
                            str_to_dx_cat["icd9code"] = str_to_dx_cat["icd9code"].str.replace(".", "", regex=False)
                            str_to_dx_cat["icd9code"] = str_to_dx_cat["icd9code"].apply(
                                lambda x: x if x in categorize_diagnosis_codes.ccs_map else icd10_to_icd9([x], [10])[0]
                            )
                            str_to_dx_cat = str_to_dx_cat[~str_to_dx_cat["icd9code"].str.contains("N/A", regex=False)]
                            str_to_dx_cat["dx_cat"] = str_to_dx_cat["icd9code"].apply(
                                lambda x: categorize_diagnosis_codes.ccs_map[x]
                            )
                            str_to_dx_cat = str_to_dx_cat.drop(columns=["icd9code"])
                            str_to_dx_cat = str_to_dx_cat.set_index("diagnosisstring").to_dict()["dx_cat"]

                            with open(os.path.join(DATA_CACHE_PATH, filename), "wb") as f:
                                pickle.dump(str_to_dx_cat, f)
                        
                        with open(os.path.join(DATA_CACHE_PATH, filename), "rb") as f:
                            str_to_dx_cat = pickle.load(f)

                        dx_code_tree = treelib.Tree()
                        dx_code_tree.create_node("root", "root")
                        for dx, cat in str_to_dx_cat.items():
                            dx = dx.split("|")
                            if not dx_code_tree.contains(dx[0]):
                                dx_code_tree.create_node(-1, dx[0], parent="root")
                            for i in range(2, len(dx)):
                                if not dx_code_tree.contains("|".join(dx[:i])):
                                    dx_code_tree.create_node(-1, "|".join(dx[:i]), parent="|".join(dx[: i - 1]))
                            if not dx_code_tree.contains("|".join(dx)):
                                dx_code_tree.create_node(cat, "|".join(dx), parent="|".join(dx[:-1]))
                        
                        nid_list = list(dx_code_tree.expand_tree(mode=treelib.Tree.DEPTH))
                        nid_list.reverse()
                        for nid in nid_list:
                            if dx_code_tree.get_node(nid).is_leaf():
                                continue
                            elif dx_code_tree.get_node(nid).tag == -1:
                                dx_code_tree.get_node(nid).tag = Counter(
                                    [child.tag for child in dx_code_tree.children(nid)]
                                ).most_common(1)[0][0]

                        single_icu_diagnosis_prediction_eicu_fn.dx_code_tree = dx_code_tree
                    
                    dx_hierarchy = dx_str.split("|")
                    for i in range(len(dx_hierarchy) - 1, 1, -1):
                        if single_icu_diagnosis_prediction_eicu_fn.dx_code_tree.contains(
                            "|".join(dx_hierarchy[:i])
                        ):
                            dx_cats_for_unmatched.append(
                                single_icu_diagnosis_prediction_eicu_fn.dx_code_tree.get_node(
                                    "|".join(dx_hierarchy[:i])
                                ).tag
                            )
                            break
                
                dx_codes.add(code)
        
        dx_codes = list(dx_codes)
        dx_categories = categorize_diagnosis_codes(dx_codes)

        if len(dx_cats_for_unmatched) > 0:
            # we don't use category number 14 since it is very rare case
            # accordingly, category number >= 15 should be shifted by 1
            dx_cats_for_unmatched = [
                x if x < 14 else x - 1 for x in list(set(dx_cats_for_unmatched)) if x != 14
            ]
            for cat in dx_cats_for_unmatched:
                dx_categories[cat] = 1

        labels[visit.visit_id] = dx_categories
    
    return labels