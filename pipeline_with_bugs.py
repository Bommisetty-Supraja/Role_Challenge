"""
Prenatal Imaging Data Pipeline
Origin Medical Research Lab -- Clinical Data Engineering

Ingests patient/study metadata and fetal biometry measurements, cleans and
normalizes key fields, merges the two sources into one analysis-ready
table, and produces a summary report + a visualization comparing clinics.

Run:
    python pipeline_with_bugs.py

Inputs (same folder):
    patient_studies.csv
    fetal_biometry.csv

Output:
    analysis_ready.csv
    clinic_summary.png
"""

import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STUDIES_PATH = "patient_studies.csv"
BIOMETRY_PATH = "fetal_biometry.csv"


def load_data(studies_path, biometry_path):
    studies = pd.read_csv(studies_path)
    biometry = pd.read_csv(biometry_path)
    return studies, biometry


def parse_notes(notes_raw):
    """
    Some `notes` values contain a JSON blob with delivery context
    (twin pregnancy, fetal presentation). Extract it into a dict;
    plain free-text notes or missing notes just return {}.
    """
    if pd.isna(notes_raw):
        return {}
    text = str(notes_raw).strip()
    if text.startswith("{") and text.endswith("}"):
        text=text.replace(r',}', '}')
        return json.loads(text)
    elif text.startswith("{") and not text.endswith("}"):
        text=text+"}"
        return json.loads(text)
    return {}


def normalize_gestational_age(ga_raw):
    """
    Convert the many gestational-age formats we receive from clinics into
    decimal weeks:
        "24w3d"    -> 24.43
        "24.4"     -> 24.4
        "26 weeks" -> 26.0
    """
    if pd.isna(ga_raw):
        return np.nan

    ga_str = str(ga_raw).strip().lower()

    match = re.match(r"(\d+)w(\d+)d", ga_str)
    if match:
        weeks, days = int(match.group(1)), int(match.group(2))
        return round(weeks + days/7,2)  

    match = re.match(r"(\d+)\s*weeks?", ga_str)
    if match:
        return round(float(match.group(1)),2)

    try:
        ga=float(ga_str)
        if ga>50:
            return round(float(ga/7.0),2)
        else:
            return round(ga,2)
    except ValueError:
        return np.nan


def normalize_device_manufacturer(name_raw):
    """Collapse manufacturer name variants down to a canonical label."""
    if pd.isna(name_raw):
        return "Unknown"
    name = str(name_raw).strip().lower()
    if "ge" in name:
        return "GE Healthcare"
    if "philips" in name:
        return "Philips"
    if "samsung" in name:
        return "Samsung"
    if "canon" in name:
        return "Canon Medical"
    return name.title()


def clean_studies(studies):
    studies = studies.copy()
    studies["gestational_age_weeks"] = studies["gestational_age_at_scan"].apply(
        normalize_gestational_age
    )
    studies["delivery_context"] = studies["notes"].apply(parse_notes)
    studies["scan_date_parsed"] = pd.to_datetime(studies["scan_date"], errors="coerce",format="mixed",utc=True).dt.date

    studies=studies.drop(columns=["scan_date","notes","gestational_age_at_scan"])

    return studies


def clean_biometry(biometry):
    biometry = biometry.copy()
    biometry["device_manufacturer_clean"] = biometry["device_manufacturer"].apply(
        normalize_device_manufacturer
    )
    biometry=biometry.drop(columns=["device_manufacturer"])
    return biometry


def merge_datasets(studies, biometry):

    studies["study_id"]= studies["study_id"].astype(str).str.upper().str.strip().str.replace("-","").str.replace("STU","STU-")
    biometry["study_id"]= biometry["study_id"].astype(str).str.upper().str.strip().str.replace("-","").str.replace("STU","STU-")

    """the patients with no fetal biometry."""
    biometry_merged = studies.merge(biometry, on="study_id", how="left")
    biometry_merged = biometry_merged[biometry_merged["bpd_mm"].isna()]
           

    """Join clinical metadata with the corresponding imaging measurements."""
    merged = studies.merge(biometry, on="study_id", how="inner")
    return merged,biometry_merged

    


def flag_high_risk(merged, clinic_thresholds):
    """
    Flag a study as high risk if its gestational age falls below the
    clinic-specific threshold for early anomaly review.

    NOTE: this currently runs nightly against ~450 studies. Ops wants to
    point it at the full historical archive (~500K studies) next quarter.
    """

    merged=merged.merge(clinic_thresholds,on="clinic_id",how="left")
    merged["review_flag"]=merged["gestational_age_weeks"]<merged["min_ga_weeks"]
    merged=merged.drop(columns=["min_ga_weeks"])
    return merged
    # flags = []
    # for i in range(len(merged)):
    #     row = merged.iloc[i]
    #     threshold = None
    #     for clinic_row in clinic_thresholds.itertuples():
    #         if clinic_row.clinic_id == row["clinic_id"]:
    #             threshold = clinic_row.min_ga_weeks
    #             break
    #     is_flagged = (
    #         pd.notna(row["gestational_age_weeks"])
    #         and threshold is not None
    #         and row["gestational_age_weeks"] < threshold
    #     )
    #     flags.append(is_flagged)
    # merged = merged.copy()
    # merged["early_review_flag"] = flags
    # return merged


def summarize_by_clinic(merged):
    summary = (
        merged.groupby("clinic_id")
        .agg(
            n_studies=("study_id", "count"),
            avg_gestational_age=("gestational_age_weeks", "mean"),
            avg_bpd_mm=("bpd_mm", "mean"),
            pct_flagged=("review_flag", "mean"),
        )
        .reset_index()
    )
    return summary


def plot_clinic_summary(summary, out_path="clinic_summary.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(summary["clinic_id"], summary["avg_gestational_age"])
    ax.set_ylabel("Avg. gestational age at scan (weeks)")
    ax.set_title("Average Gestational Age at Scan by Clinic")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_biometry_missing(biometry_merged,out_path="fetal_biometry_missing.png"):
    fig,ax=plt.subplots(figsize=(8,5))
    count_for_clinic=biometry_merged.groupby("clinic_id").agg(patient_count=("study_id","count")).reset_index()
    ax.bar(count_for_clinic["clinic_id"],count_for_clinic["patient_count"],color="blue")
    ax.set_yticks(range(0,count_for_clinic["patient_count"].max()+1,1))
    ax.set_xlabel("Clinic ID")
    ax.set_ylabel("Number of Patients with Missing Biometry")
    ax.set_title("Patients with Missing Biometry for each Clinic")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_meternal_risk(merged, out_path="meternity_risk_identity.png"):
    df=merged.copy()
    rules=[
        (df["maternal_age"]<20) | (df["maternal_age"]>=40),
        (df["maternal_age"]>=35) & (df["maternal_age"]<=39),
        (df["maternal_age"]>20) & (df["maternal_age"]<=34)
    ]
    df["severity_level"]=np.select(rules,["High_Risk","Mid_Risk","Low_Risk"],default="Unknown")
    df=df[df["severity_level"]!="Unknown"]
    severity_count=df.groupby(["clinic_id","severity_level"]).size().unstack(fill_value=0)
    fig,ax=plt.subplots(figsize=(10,6))
    severity_count.plot(kind="bar", ax=ax,color=["red","orange","green"])
    ax.set_xlabel("clinic_id")
    ax.set_ylabel("Patients count")
    ax.legend(title="age based severity level")
    ax.set_title("Patients count by severity level for each clinic")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def main():
    studies, biometry = load_data(STUDIES_PATH, BIOMETRY_PATH)
    print(f"Loaded {len(studies)} study records and {len(biometry)} biometry records.")

    studies=studies.drop_duplicates()
    biometry=biometry.drop_duplicates()


    studies = clean_studies(studies)
    biometry = clean_biometry(biometry)

    print("After Deduplication")
    print(f"Loaded {len(studies)} study records and {len(biometry)} biometry records.")

    # studies.to_csv("cleanedStudies.csv",index=False)
    # biometry.to_csv("cleanedBiometry.csv",index=False)

    # duplicates=studies[studies["study_id"].duplicated(keep=False)].sort_values("study_id")
    # print(duplicates)

    # duplicates_rows=biometry[biometry["study_id"].duplicated(keep=False)].sort_values("study_id")
    # print(duplicates_rows)


    merged, biometry_merged = merge_datasets(studies, biometry)
    print(f"Merged dataset has {len(merged)} rows.")

    clinic_thresholds = pd.DataFrame(
        {
            "clinic_id": ["CLN-01", "CLN-02", "CLN-03", "CLN-04", "CLN-05", "CLN-06", "CLN-07"],
            "min_ga_weeks": [18, 20, 18, 22, 20, 18, 20],
        }
    )
    merged = flag_high_risk(merged, clinic_thresholds)

    summary = summarize_by_clinic(merged)
    print("\nClinic summary:")
    print(summary.to_string(index=False))

    biometry_merged.to_csv("patients_with_missing_biometry.csv", index=False)
    merged.to_csv("analysis_ready.csv", index=False)
    plot_clinic_summary(summary)
    plot_biometry_missing(biometry_merged)
    plot_meternal_risk(merged)
    print("\nWrote analysis_ready.csv, clinic_summary.png, fetal_biometry_missing.png and meternity_risk_identity.png")


if __name__ == "__main__":
    main()
