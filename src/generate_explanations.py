import sys
import os
import json
import hashlib
import time
import re
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))

from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL

client = Groq(api_key=GROQ_API_KEY)
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

NOVICE_TEMPLATE = (
    "You applied for credit and a machine learning model assessed your application. "
    "Here are the factors that influenced the decision about you (feature: SHAP value):\n{shap_summary}\n\n"
    "Explain this decision directly to the applicant as if they have no finance or technical background. "
    "Use plain English and everyday analogies. No numbers, no jargon. Keep it under 150 words."
)

EXPERT_TEMPLATE = (
    "You applied for credit and a machine learning model assessed your application. "
    "Here are the SHAP values for each feature in your case:\n{shap_summary}\n\n"
    "Explain this decision directly to the applicant as if they have a data science background. "
    "Reference the exact feature names and their SHAP values. "
    "Be precise about direction and magnitude. Keep it under 150 words."
)

INCOME_NOVICE_TEMPLATE = (
    "A machine learning model predicted your income level based on your personal and professional details. "
    "Here are the factors that influenced the prediction about you (feature: SHAP value):\n{shap_summary}\n\n"
    "Explain this prediction directly to the person as if they have no finance or technical background. "
    "Use plain English and everyday analogies. No numbers, no jargon. Keep it under 150 words."
)

INCOME_EXPERT_TEMPLATE = (
    "A machine learning model predicted your income level based on your personal and professional details. "
    "Here are the SHAP values for each feature in your case:\n{shap_summary}\n\n"
    "Explain this prediction directly to the person as if they have a data science background. "
    "Reference the exact feature names and their SHAP values. "
    "Be precise about direction and magnitude. Keep it under 150 words."
)


def cache_key(instance_id, persona, dataset):
    raw = f"{dataset}_{instance_id}_{persona}"
    return hashlib.md5(raw.encode()).hexdigest()


def load_from_cache(instance_id, persona, dataset):
    key = cache_key(instance_id, persona, dataset)
    path = os.path.join(CACHE_DIR, f"{key}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)["explanation"]
    return None


def save_to_cache(instance_id, persona, dataset, explanation):
    key = cache_key(instance_id, persona, dataset)
    path = os.path.join(CACHE_DIR, f"{key}.json")
    with open(path, "w") as f:
        json.dump({"explanation": explanation}, f)


def build_shap_summary(feature_names, shap_row, top_n=5):
    pairs = sorted(zip(feature_names, shap_row), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    return "\n".join(f"  {feat}: {val:+.4f}" for feat, val in pairs)


def call_groq(prompt):
    while True:
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate_limit" in msg.lower():
                match = re.search(r"try again in (\d+)m(\d+(?:\.\d+)?)s", msg)
                if match:
                    wait = int(match.group(1)) * 60 + float(match.group(2)) + 1
                else:
                    wait = 60
                print(f"Rate limit hit — waiting {wait:.0f}s...")
                time.sleep(wait)
            else:
                raise


def generate_for_dataset(shap_path, feature_path, output_path, dataset_name,
                          novice_tmpl, expert_tmpl, max_instances=150):
    shap_df = pd.read_csv(shap_path)
    feature_names = pd.read_csv(feature_path).columns.tolist()
    shap_df = shap_df.iloc[:max_instances]

    rows = []
    for i, shap_row in enumerate(shap_df.values):
        shap_summary = build_shap_summary(feature_names, shap_row)

        for persona, tmpl in [("Novice", novice_tmpl), ("Expert", expert_tmpl)]:
            cached = load_from_cache(i, persona, dataset_name)
            if cached:
                explanation = cached
            else:
                prompt = tmpl.format(shap_summary=shap_summary)
                explanation = call_groq(prompt)
                save_to_cache(i, persona, dataset_name, explanation)
            rows.append({"instance_id": i, "persona": persona, "explanation": explanation})

        if (i + 1) % 10 == 0:
            print(f"[{dataset_name}] Processed {i + 1}/{len(shap_df)} instances")
            pd.DataFrame(rows).to_csv(output_path, index=False)

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"[{dataset_name}] Saved {len(rows)} explanations to {output_path}")


if __name__ == "__main__":
    base = os.path.join(os.path.dirname(__file__), "..")

    generate_for_dataset(
        shap_path=os.path.join(base, "data", "german_shap_values.csv"),
        feature_path=os.path.join(base, "data", "german_X_test.csv"),
        output_path=os.path.join(base, "data", "german_explanations.csv"),
        dataset_name="german",
        novice_tmpl=NOVICE_TEMPLATE,
        expert_tmpl=EXPERT_TEMPLATE,
        max_instances=150,
    )

    generate_for_dataset(
        shap_path=os.path.join(base, "data", "adult_shap_values.csv"),
        feature_path=os.path.join(base, "data", "adult_X_test.csv"),
        output_path=os.path.join(base, "data", "adult_explanations.csv"),
        dataset_name="adult",
        novice_tmpl=INCOME_NOVICE_TEMPLATE,
        expert_tmpl=INCOME_EXPERT_TEMPLATE,
        max_instances=150,
    )
