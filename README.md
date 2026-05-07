# Fincast

Live stock dashboard built with Streamlit, Yahoo Finance, XGBoost, and a Q-learning policy replay.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

1. Push this folder to its own GitHub repository.
2. In Streamlit Community Cloud, create a new app from that repository.
3. Set the main file path to `app.py`.
4. In the app settings, add this secret:

```toml
NEWS_API_KEY = "your_newsapi_key_here"
```

## Automated Refresh

The GitHub Actions workflow in `.github/workflows/update.yml` can refresh price data and retrain model artifacts daily after the repository is pushed to GitHub.
