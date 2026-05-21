import anthropic
from src.config import API_KEY, MODEL


def generate_explanation(shap_values, feature_names, instance_index=0):
    client = anthropic.Anthropic(api_key=API_KEY)
    shap_row = shap_values[instance_index]
    top_features = sorted(
        zip(feature_names, shap_row), key=lambda x: abs(x[1]), reverse=True
    )[:5]
    feature_str = "\n".join(f"  {name}: {val:.4f}" for name, val in top_features)
    prompt = (
        f"The following SHAP values explain a model prediction. "
        f"Positive values push toward the positive class, negative away.\n\n"
        f"Top features:\n{feature_str}\n\n"
        f"Write a concise, plain-English explanation of why the model made this prediction."
    )
    message = client.messages.create(
        model=MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text
