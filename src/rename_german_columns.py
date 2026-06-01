import pandas as pd

RENAME_MAP = {
    "Attribute2": "duration_months",
    "Attribute5": "credit_amount",
    "Attribute8": "installment_rate_pct",
    "Attribute11": "residence_since_years",
    "Attribute13": "age_years",
    "Attribute16": "existing_credits_count",
    "Attribute18": "dependents_count",
    "Attribute1_A12": "checking_0_to_200DM",
    "Attribute1_A13": "checking_over_200DM",
    "Attribute1_A14": "checking_no_account",
    "Attribute3_A31": "credit_history_all_paid_here",
    "Attribute3_A32": "credit_history_existing_paid",
    "Attribute3_A33": "credit_history_delays_past",
    "Attribute3_A34": "credit_history_critical_other_credits",
    "Attribute4_A41": "purpose_used_car",
    "Attribute4_A410": "purpose_other",
    "Attribute4_A42": "purpose_furniture",
    "Attribute4_A43": "purpose_radio_tv",
    "Attribute4_A44": "purpose_appliances",
    "Attribute4_A45": "purpose_repairs",
    "Attribute4_A46": "purpose_education",
    "Attribute4_A48": "purpose_retraining",
    "Attribute4_A49": "purpose_business",
    "Attribute6_A62": "savings_100_to_500DM",
    "Attribute6_A63": "savings_500_to_1000DM",
    "Attribute6_A64": "savings_over_1000DM",
    "Attribute6_A65": "savings_no_account",
    "Attribute7_A72": "employment_under_1yr",
    "Attribute7_A73": "employment_1_to_4yrs",
    "Attribute7_A74": "employment_4_to_7yrs",
    "Attribute7_A75": "employment_over_7yrs",
    "Attribute9_A92": "female_or_married",
    "Attribute9_A93": "male_single",
    "Attribute9_A94": "male_married_widowed",
    "Attribute10_A102": "has_co_applicant",
    "Attribute10_A103": "has_guarantor",
    "Attribute12_A122": "property_building_society",
    "Attribute12_A123": "property_car_or_other",
    "Attribute12_A124": "property_none",
    "Attribute14_A142": "other_installments_stores",
    "Attribute14_A143": "other_installments_none",
    "Attribute15_A152": "housing_own",
    "Attribute15_A153": "housing_free",
    "Attribute17_A172": "job_unskilled_resident",
    "Attribute17_A173": "job_skilled",
    "Attribute17_A174": "job_highly_skilled",
    "Attribute19_A192": "has_telephone",
    "Attribute20_A202": "not_foreign_worker",
}

FILES = [
    "data/german_X_train.csv",
    "data/german_X_test.csv",
    "data/german_shap_values.csv",
]

for path in FILES:
    df = pd.read_csv(path)
    df.rename(columns=RENAME_MAP, inplace=True)
    df.to_csv(path, index=False)
    print(f"renamed: {path}")
    print(f"  first 5 columns: {list(df.columns[:5])}")
