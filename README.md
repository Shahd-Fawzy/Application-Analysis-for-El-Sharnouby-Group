# EL-Sharnouby Decision Intelligence (Python + Streamlit)

A **local, runnable** companion to the Power BI dashboards: forecasting, stock replenishment, scenario planning, customer retention, factory and printing performance, marketing/procurement analysis, and a management action center.

## How to run

Requires **Python 3.10+** and an internet connection for the *initial package installation only*:

**Windows PowerShell**
```powershell
cd EL_Sharnouby_Python_App
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

**macOS / Linux**
```bash
cd EL_Sharnouby_Python_App
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the localhost URL printed by Streamlit (usually http://localhost:8501). The app can run offline after dependencies have been installed.

## Pages

1. Executive overview
2. Demand forecasting (rolling historical backtest, 30/60/90-day approximation)
3. Inventory optimizer (user-controlled days of coverage and lead-time scenario)
4. What-if business simulator (user-chosen assumptions)
5. Customer RFM and cohort retention
6. Factory, printing and quality risks
7. Marketing and supplier comparison
8. Decision action center (CSV export)
9. Methods and limitations

## Validation

Run `python -m unittest discover -s tests -v` to execute smoke/data consistency tests.

## Important

The original challenge dataset contains **modeled training transactions** alongside real/public master data. Forecasts, risks, savings and recommendations are not audited business outcomes. Most analyses are deterministic, transparent baselines/heuristics; the app must not be sold as a trained causal AI engine or a verified digital twin. All financial amounts are labeled EGP. The separate Power BI report is not modified by this app.

Records are packaged in `data/`, no re-upload required. The PDF challenge brief is not included to avoid a ~40 MB duplicate.
