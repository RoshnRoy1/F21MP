# XAI Dissertation

Explainable AI research comparing SHAP values with LLM-generated textual explanations.

## Structure

```
xai_dissertation/
├── src/
│   ├── config.py               # env + model config
│   ├── preprocess.py           # data cleaning / encoding
│   ├── train_shap.py           # XGBoost training + SHAP computation
│   ├── generate_explanations.py # Claude-powered natural language explanations
│   └── textual_shapgap.py      # TextualSHAPGAP metric
├── data/                       # raw CSV datasets (gitignored)
├── cache/                      # cached SHAP / API results (gitignored)
├── notebooks/                  # Jupyter exploration
├── .env                        # API keys (gitignored)
└── requirements.txt
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Datasets

- German Credit (UCI id=144)
- Adult Income (UCI id=2)
