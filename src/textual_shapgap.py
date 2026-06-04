import os
import re
import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), "..")

# ---------------------------------------------------------------------------
# Direction and rank detection
# ---------------------------------------------------------------------------

POS_RE = re.compile(
    r'\b(positive|positively|increas\w*|boost\w*|help\w*|good|better|higher|support\w*|'
    r'benefit\w*|favorable|favor\w*|strong\w*|strength|healthy|improv\w*|approv\w*|'
    r'advantage\w*|contributes?\s+positively|likelihood\s+of\s+approval|'
    r'positive\s+effect|positive\s+impact|positive\s+influence|encouraging|promising)\b',
    re.I,
)
NEG_RE = re.compile(
    r'\b(negative|negatively|decreas\w*|lower\w*|hurt\w*|bad|worse|risk\w*|harm\w*|'
    r'unfavorable|weak\w*|poor|concern\w*|penaliz\w*|penalty|against|red\s+flag|'
    r'disadvantage\w*|reduces?\s+the\s+likelihood|detracts?\w*|contributes?\s+negatively|'
    r'negative\s+effect|negative\s+impact|uncertain\w*|worr\w*|alarming|too\s+much|'
    r'difficult\w*|problem\w*|issue\w*|obstacle\w*)\b',
    re.I,
)
NEGATION_RE = re.compile(
    r"\b(not|no|without|lack\w*|doesn'?t|don'?t|didn'?t|never|none|isn'?t|wasn'?t|aren'?t|hardly)\b",
    re.I,
)

RANK_PATTERNS = [
    (1, re.compile(
        r'\b(most\s+important|biggest\s+factor|primary|main\s+factor|strongest|greatest|'
        r'dominant|largest\s+impact|most\s+significant|most\s+influential|key\s+factor|'
        r'most\s+impactful|the\s+largest|highest\s+impact|primarily|much\s+larger|'
        r'had\s+a\s+much|the\s+strongest|strong\s+negative|strong\s+positive|'
        r'most\s+strongly|heavily\s+influenced|the\s+biggest|greatest\s+influence|'
        r'the\s+former|former\s+having|much\s+larger\s+impact)\b',
        re.I,
    )),
    (2, re.compile(
        r'\b(second\w*|also\s+significant|another\s+major|secondary|also\s+notably|'
        r'also\s+strongly)\b',
        re.I,
    )),
    (3, re.compile(r'\b(third|also\s+notable|third\s+factor)\b', re.I)),
    (4, re.compile(r'\b(fourth|minor\s+factor)\b', re.I)),
    (5, re.compile(
        r'\b(fifth|slight\w*|minimal|least\s+important|small\s+effect|small\s+positive|'
        r'small\s+negative|minor\s+effect|minor\s+positive|minor\s+negative)\b',
        re.I,
    )),
]


def detect_direction(text: str, pos: int, window: int = 150) -> int | None:
    start = max(0, pos - window)
    end = min(len(text), pos + window)
    snippet = text[start:end]
    pos_hits = len(POS_RE.findall(snippet))
    neg_hits = len(NEG_RE.findall(snippet))
    neg_words = len(NEGATION_RE.findall(snippet))
    # Negation near positive language flips the reading
    if neg_words >= 1 and pos_hits > neg_hits:
        pos_hits, neg_hits = 0, neg_hits + 1
    if pos_hits > neg_hits:
        return 1
    if neg_hits > pos_hits:
        return -1
    return None


def detect_rank(text: str, pos: int, window: int = 350) -> int | None:
    start = max(0, pos - window)
    end = min(len(text), pos + window)
    snippet = text[start:end]
    for rank, pat in RANK_PATTERNS:
        if pat.search(snippet):
            return rank
    return None


def find_in_text(text: str, aliases: list) -> tuple:
    lower = text.lower()
    for alias in sorted(aliases, key=len, reverse=True):  # longest first
        idx = lower.find(alias.lower())
        if idx != -1:
            return idx, alias
    return None, None


# ---------------------------------------------------------------------------
# Synonym maps
# ---------------------------------------------------------------------------

GERMAN_SYNONYMS = {
    "duration_months": ["duration", "months", "loan period", "loan term", "term",
                        "loan duration", "length of loan", "how long"],
    "credit_amount": ["credit amount", "loan amount", "loan size", "amount of credit",
                      "amount borrowed", "amount", "credit"],
    "installment_rate_pct": ["installment rate", "payment rate", "repayment rate",
                              "installment", "monthly payment percentage"],
    "residence_since_years": ["residence", "years at address", "current address",
                               "living at", "time at current", "present residence"],
    "age_years": ["age", "years old", "older", "younger", "old", "young"],
    "existing_credits_count": ["existing credits", "number of credits", "credits at this bank",
                                "existing loans", "multiple credits", "credit count",
                                "existing credit", "multiple existing"],
    "dependents_count": ["dependents", "people liable", "maintenance", "supporting",
                         "liable for", "number of people"],
    "checking_0_to_200DM": ["checking between 0 and 200", "checking 0 to 200",
                              "small checking balance", "low checking balance"],
    "checking_over_200DM": ["checking over 200", "checking above 200",
                             "good checking balance", "positive checking"],
    "checking_no_account": ["no checking account", "no checking", "without a checking",
                             "without an account", "no account", "checking account"],
    "credit_history_all_paid_here": ["all credits paid", "clean credit history",
                                      "good credit history", "paid back duly",
                                      "fully paid", "all paid here"],
    "credit_history_existing_paid": ["existing credits paid", "credits paid duly",
                                      "currently paying on time", "existing paid"],
    "credit_history_delays_past": ["delays", "delay", "late payment", "payment delays",
                                    "past delays", "delayed payment", "previously delayed"],
    "credit_history_critical_other_credits": ["critical account", "other credits",
                                               "credit elsewhere", "credit at other banks",
                                               "external credit", "critical credit"],
    "purpose_used_car": ["used car", "second-hand car", "secondhand car",
                          "pre-owned car", "used vehicle"],
    "purpose_other": ["other purpose", "other loan purpose"],
    "purpose_furniture": ["furniture", "furnishing", "home furnishing"],
    "purpose_radio_tv": ["radio", "television", "tv", "electronics",
                          "entertainment appliance"],
    "purpose_appliances": ["appliances", "domestic appliances", "household appliances"],
    "purpose_repairs": ["repairs", "repair", "home repair", "fixing"],
    "purpose_education": ["education", "school", "studying", "university",
                           "college", "learning"],
    "purpose_retraining": ["retraining", "re-training", "job training",
                            "vocational training"],
    "purpose_business": ["business", "entrepreneurship", "commercial",
                          "business purpose"],
    "savings_100_to_500DM": ["savings between 100 and 500", "savings 100 to 500",
                               "small savings", "modest savings"],
    "savings_500_to_1000DM": ["savings between 500 and 1000", "savings 500 to 1000",
                                "decent savings", "moderate savings"],
    "savings_over_1000DM": ["savings over 1000", "over 1000 dm", "substantial savings",
                             "significant savings", "large savings", "high savings",
                             "1000 dm", "savings over", "abundant savings",
                             "savings", "saved"],
    "savings_no_account": ["no savings account", "no savings", "without savings",
                            "no saving", "no money saved"],
    "employment_under_1yr": ["employment under 1 year", "under 1 year",
                              "less than a year", "new job", "recently employed",
                              "employment under", "under one year", "short employment",
                              "employment for less"],
    "employment_1_to_4yrs": ["employment 1 to 4", "1 to 4 years",
                               "one to four years", "a few years employed"],
    "employment_4_to_7yrs": ["employment 4 to 7", "4 to 7 years",
                               "four to seven years", "several years employed"],
    "employment_over_7yrs": ["employment over 7", "over 7 years",
                              "more than 7 years", "long-term employment",
                              "stable employment", "long employment",
                              "many years employed", "over seven years"],
    "female_or_married": ["female", "woman", "married woman", "personal status", "gender"],
    "male_single": ["male single", "single male", "unmarried male", "single man"],
    "male_married_widowed": ["married male", "widowed male", "married man", "widowed man"],
    "has_co_applicant": ["co-applicant", "co applicant", "joint applicant",
                          "co-signer", "joint application"],
    "has_guarantor": ["guarantor", "guarantee", "backed by guarantor", "guaranteeing"],
    "property_building_society": ["building society", "life insurance",
                                   "savings scheme", "building savings"],
    "property_car_or_other": ["car or other property", "property car",
                               "vehicle property"],
    "property_none": ["no property", "no assets", "nothing to offer",
                       "without property", "unknown property"],
    "other_installments_stores": ["installment plans at stores", "store credit",
                                   "other installments at stores", "store installment",
                                   "other payment plans at stores"],
    "other_installments_none": ["no other installments", "no installment plans",
                                  "no other payment plans", "no other loans",
                                  "no other obligations", "without other installments",
                                  "no other debts", "other installments"],
    "housing_own": ["own home", "homeowner", "own house", "owns their home",
                    "home owner", "property owner", "owns a home", "own property"],
    "housing_free": ["free housing", "rent-free", "lives for free",
                     "free accommodation"],
    "job_unskilled_resident": ["unskilled worker", "unskilled job",
                                "unskilled resident", "low-skilled", "manual labor"],
    "job_skilled": ["skilled worker", "skilled employee", "skilled job",
                    "skilled occupation", "skilled official", "skilled", "official"],
    "job_highly_skilled": ["highly skilled", "management", "self-employed",
                            "manager", "professional", "highly qualified",
                            "executive", "officer"],
    "has_telephone": ["telephone", "phone", "registered phone", "landline",
                      "has a phone"],
    "not_foreign_worker": ["not a foreign worker", "domestic worker",
                            "foreign worker", "not foreign", "nationality",
                            "citizen", "foreign"],
}

ADULT_SYNONYMS = {
    "age": ["age", "years old", "older", "younger", "how old", "their age"],
    "fnlwgt": ["fnlwgt", "sampling weight", "census weight", "statistical weight",
                "final weight", "population weight"],
    "education-num": ["education-num", "education num", "years of education",
                       "educational level", "education level", "schooling",
                       "years in school"],
    "capital-gain": ["capital-gain", "capital gain", "investment gain",
                      "investment profit", "gains from investment",
                      "capital gains", "investment income", "capital"],
    "capital-loss": ["capital-loss", "capital loss", "investment loss",
                      "financial loss", "investment losses", "losses from investment"],
    "hours-per-week": ["hours-per-week", "hours per week", "working hours",
                        "work hours", "hours worked", "weekly hours",
                        "overtime", "work schedule"],
    "workclass_Local-gov": ["local government", "local gov", "local-gov"],
    "workclass_Never-worked": ["never worked", "never-worked"],
    "workclass_Private": ["private sector", "private company",
                           "private employer", "private"],
    "workclass_Self-emp-inc": ["self-employed incorporated", "self-emp-inc",
                                "incorporated self-employed"],
    "workclass_Self-emp-not-inc": ["self-employed", "self employed",
                                    "own business", "self-emp"],
    "workclass_State-gov": ["state government", "state gov", "state-gov"],
    "workclass_Without-pay": ["without pay", "volunteer", "unpaid"],
    "education_11th": ["11th grade", "11th"],
    "education_12th": ["12th grade", "12th"],
    "education_1st-4th": ["1st to 4th grade", "elementary school", "primary school"],
    "education_5th-6th": ["5th to 6th grade"],
    "education_7th-8th": ["7th to 8th grade", "middle school"],
    "education_9th": ["9th grade"],
    "education_Assoc-acdm": ["associate academic", "assoc-acdm",
                               "academic associate degree"],
    "education_Assoc-voc": ["associate vocational", "assoc-voc",
                              "vocational associate"],
    "education_Bachelors": ["bachelor", "bachelors", "bachelors degree",
                             "undergraduate degree", "college degree"],
    "education_Doctorate": ["doctorate", "phd", "doctoral degree",
                             "doctor of philosophy"],
    "education_HS-grad": ["high school graduate", "hs-grad", "high school diploma",
                           "high school grad", "hs grad"],
    "education_Masters": ["masters", "masters degree", "master's degree",
                           "graduate degree"],
    "education_Preschool": ["preschool", "pre-school"],
    "education_Prof-school": ["professional school", "prof-school",
                               "law school", "medical school"],
    "education_Some-college": ["some college", "partial college", "some-college"],
    "marital-status_Married-AF-spouse": ["married armed forces", "af spouse",
                                          "military spouse", "armed forces spouse"],
    "marital-status_Married-civ-spouse": ["married-civ-spouse", "married civ spouse",
                                           "civil spouse", "married civilian",
                                           "married", "spouse", "civil marriage"],
    "marital-status_Married-spouse-absent": ["spouse absent", "spouse-absent",
                                              "absent spouse"],
    "marital-status_Never-married": ["never married", "never-married",
                                      "unmarried", "never been married"],
    "marital-status_Separated": ["separated", "separation"],
    "marital-status_Widowed": ["widowed", "widow", "widower"],
    "occupation_Armed-Forces": ["armed forces", "military", "army", "armed-forces"],
    "occupation_Craft-repair": ["craft repair", "craft-repair", "craftsman",
                                 "repair work", "skilled trade"],
    "occupation_Exec-managerial": ["exec-managerial", "executive", "managerial",
                                    "management", "manager", "executive position",
                                    "exec managerial"],
    "occupation_Farming-fishing": ["farming", "fishing", "agriculture",
                                    "farm", "farming-fishing"],
    "occupation_Handlers-cleaners": ["handlers", "cleaners", "handler",
                                      "cleaner", "handlers-cleaners"],
    "occupation_Machine-op-inspct": ["machine-op-inspct", "machine operator",
                                      "machine inspection", "machine op"],
    "occupation_Other-service": ["other-service", "other service",
                                  "service worker", "service industry"],
    "occupation_Priv-house-serv": ["private household", "house service",
                                    "domestic service", "priv-house-serv"],
    "occupation_Prof-specialty": ["prof-specialty", "professional specialty",
                                   "professional job", "professional occupation",
                                   "specialist", "prof specialty"],
    "occupation_Protective-serv": ["protective service", "protective-serv",
                                    "police", "security", "law enforcement"],
    "occupation_Sales": ["sales", "sales job", "selling", "salesperson"],
    "occupation_Tech-support": ["tech-support", "tech support",
                                 "technical support", "technology support"],
    "occupation_Transport-moving": ["transport", "moving", "transportation",
                                     "transport-moving", "driver", "delivery"],
    "relationship_Not-in-family": ["not-in-family", "not in family",
                                    "outside family"],
    "relationship_Other-relative": ["other-relative", "other relative", "relative"],
    "relationship_Own-child": ["own-child", "own child", "child", "son", "daughter"],
    "relationship_Unmarried": ["unmarried relationship", "not in relationship"],
    "relationship_Wife": ["wife", "spouse", "married partner"],
    "race_Asian-Pac-Islander": ["asian", "pacific islander", "asian-pac-islander"],
    "race_Black": ["black", "african american", "african-american"],
    "race_Other": ["other race", "mixed race"],
    "race_White": ["white", "caucasian"],
    "sex_Male": ["male", "man", "masculine", "being male", "sex male", "gender male"],
    "native-country_Canada": ["canada", "canadian"],
    "native-country_China": ["china", "chinese"],
    "native-country_Columbia": ["colombia", "colombian", "columbia"],
    "native-country_Cuba": ["cuba", "cuban"],
    "native-country_Dominican-Republic": ["dominican republic", "dominican"],
    "native-country_Ecuador": ["ecuador", "ecuadorian"],
    "native-country_El-Salvador": ["el salvador", "salvadoran"],
    "native-country_England": ["england", "english", "british"],
    "native-country_France": ["france", "french"],
    "native-country_Germany": ["germany", "german"],
    "native-country_Greece": ["greece", "greek"],
    "native-country_Guatemala": ["guatemala", "guatemalan"],
    "native-country_Haiti": ["haiti", "haitian"],
    "native-country_Holand-Netherlands": ["netherlands", "dutch", "holland"],
    "native-country_Honduras": ["honduras", "honduran"],
    "native-country_Hong": ["hong kong"],
    "native-country_Hungary": ["hungary", "hungarian"],
    "native-country_India": ["india", "indian"],
    "native-country_Iran": ["iran", "iranian", "persian"],
    "native-country_Ireland": ["ireland", "irish"],
    "native-country_Italy": ["italy", "italian"],
    "native-country_Jamaica": ["jamaica", "jamaican"],
    "native-country_Japan": ["japan", "japanese"],
    "native-country_Laos": ["laos", "laotian"],
    "native-country_Mexico": ["mexico", "mexican"],
    "native-country_Nicaragua": ["nicaragua", "nicaraguan"],
    "native-country_Outlying-US(Guam-USVI-etc)": ["guam", "outlying us", "us territories"],
    "native-country_Peru": ["peru", "peruvian"],
    "native-country_Philippines": ["philippines", "filipino"],
    "native-country_Poland": ["poland", "polish"],
    "native-country_Portugal": ["portugal", "portuguese"],
    "native-country_Puerto-Rico": ["puerto rico", "puerto rican"],
    "native-country_Scotland": ["scotland", "scottish"],
    "native-country_South": ["south korea", "south", "korea"],
    "native-country_Taiwan": ["taiwan", "taiwanese"],
    "native-country_Thailand": ["thailand", "thai"],
    "native-country_Trinadad&Tobago": ["trinidad", "tobago", "trinidad and tobago"],
    "native-country_United-States": ["united states", "america", "american", "usa"],
    "native-country_Vietnam": ["vietnam", "vietnamese"],
    "native-country_Yugoslavia": ["yugoslavia", "yugoslav"],
}


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_vector(text: str, feature_names: list, synonym_map: dict,
                   shap_row: np.ndarray) -> np.ndarray:
    vec = np.zeros(len(feature_names))
    feat_idx = {f: i for i, f in enumerate(feature_names)}

    top5_indices = np.argsort(np.abs(shap_row))[::-1][:5]
    top5_features = [feature_names[i] for i in top5_indices]

    mentioned = {}  # feat -> (direction, rank)

    for feat in top5_features:
        base_aliases = synonym_map.get(feat, [])
        extra = [feat, feat.replace("_", " "), feat.replace("-", " ")]
        aliases = list(dict.fromkeys(base_aliases + extra))

        pos, _ = find_in_text(text, aliases)
        if pos is None:
            continue

        direction = detect_direction(text, pos)
        if direction is None:
            direction = 1 if shap_row[feat_idx[feat]] >= 0 else -1

        rank = detect_rank(text, pos)
        mentioned[feat] = (direction, rank)

    if not mentioned:
        return vec

    has_explicit_rank = any(r is not None for _, r in mentioned.values())

    for feat, (direction, rank) in mentioned.items():
        weight = (6 - rank) if (has_explicit_rank and rank is not None) else 1.0
        vec[feat_idx[feat]] = direction * weight

    return vec


# ---------------------------------------------------------------------------
# Dataset processing
# ---------------------------------------------------------------------------

def process_dataset(explanations_path: str, shap_path: str, feature_path: str,
                    synonym_map: dict, output_path: str, name: str) -> None:
    expl_df = pd.read_csv(explanations_path)
    shap_df = pd.read_csv(shap_path)
    feature_names = pd.read_csv(feature_path).columns.tolist()

    results = []

    for _, row in expl_df.iterrows():
        instance_id = int(row["instance_id"])
        persona = row["persona"]
        text = str(row["explanation"])

        shap_row = shap_df.iloc[instance_id].values.astype(float)
        extracted = extract_vector(text, feature_names, synonym_map, shap_row)

        # MSE before normalisation
        mse = float(np.mean((shap_row - extracted) ** 2))

        # Cosine similarity on unit-length vectors
        na = np.linalg.norm(shap_row)
        nb = np.linalg.norm(extracted)
        cos_sim = float(np.dot(shap_row / na, extracted / nb)) if (na > 1e-10 and nb > 1e-10) else 0.0

        results.append({
            "instance_id": instance_id,
            "persona": persona,
            "cosine_similarity": cos_sim,
            "mse": mse,
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, index=False)
    print(f"\n[{name}] Saved {len(out_df)} rows → {output_path}")
    for persona in ["Novice", "Expert"]:
        sub = out_df[out_df["persona"] == persona]
        print(f"  {persona:6s}  mean_cosine={sub['cosine_similarity'].mean():.4f}"
              f"  mean_mse={sub['mse'].mean():.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    process_dataset(
        explanations_path=os.path.join(BASE, "data", "german_explanations.csv"),
        shap_path=os.path.join(BASE, "data", "german_shap_values.csv"),
        feature_path=os.path.join(BASE, "data", "german_X_test.csv"),
        synonym_map=GERMAN_SYNONYMS,
        output_path=os.path.join(BASE, "data", "german_shapgap_scores.csv"),
        name="german",
    )

    process_dataset(
        explanations_path=os.path.join(BASE, "data", "adult_explanations.csv"),
        shap_path=os.path.join(BASE, "data", "adult_shap_values.csv"),
        feature_path=os.path.join(BASE, "data", "adult_X_test.csv"),
        synonym_map=ADULT_SYNONYMS,
        output_path=os.path.join(BASE, "data", "adult_shapgap_scores.csv"),
        name="adult",
    )

    print("\nTextual ShapGAP complete")
