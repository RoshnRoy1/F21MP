"""
TextualSHAPGAP: measures the gap between SHAP-ranked feature importance
and the importance implied by LLM-generated textual explanations.
"""

import re
from collections import defaultdict


def rank_features_by_shap(shap_values, feature_names):
    mean_abs = abs(shap_values).mean(axis=0)
    ranked = sorted(zip(feature_names, mean_abs), key=lambda x: x[1], reverse=True)
    return [name for name, _ in ranked]


def rank_features_in_text(text, feature_names):
    text_lower = text.lower()
    mentions = defaultdict(int)
    for name in feature_names:
        mentions[name] = len(re.findall(re.escape(name.lower()), text_lower))
    ranked = sorted(feature_names, key=lambda n: mentions[n], reverse=True)
    return ranked


def shapgap_score(shap_rank, text_rank):
    n = len(shap_rank)
    if n == 0:
        return 0.0
    total = sum(abs(shap_rank.index(f) - text_rank.index(f)) for f in shap_rank if f in text_rank)
    max_gap = n * (n - 1) / 2
    return 1 - total / max_gap if max_gap > 0 else 1.0
