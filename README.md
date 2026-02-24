# ⚽ Football Predictor

An AI-powered football match prediction engine focused on finding **value bets** — where bookmaker odds are mispriced compared to our model's probability estimates.

## 🎯 Approach

Rather than just predicting winners, we model match outcome probabilities and compare them against bookmaker implied probabilities. A bet only has value when our model gives a higher probability than the market.

## 🗂️ Project Structure

```
football-predictor/
├── data/
│   └── raw/          # Raw CSVs from football-data.co.uk
├── notebooks/        # Exploratory analysis
├── src/
│   ├── data_loader.py   # Data fetching & storage
│   ├── features.py      # Feature engineering
│   └── model.py         # ML model training & prediction
├── requirements.txt
└── README.md
```

## 📦 Data Sources

- **[football-data.co.uk](https://www.football-data.co.uk/)** — Historical match results + odds (free)
- **[Understat](https://understat.com/)** — xG data for top European leagues
- **[API-Football](https://www.api-football.com/)** — Live stats, lineups, injuries

## 🛠️ Tech Stack

- Python 3.13
- pandas, numpy — data processing
- scikit-learn, XGBoost — ML models
- matplotlib — visualisation
- Jupyter — exploration

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/Jonathannd03/football-predictor.git
cd football-predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Load first dataset
python src/data_loader.py
```

## 📍 Roadmap

- [x] Project setup & data pipeline
- [ ] Feature engineering (form, H2H, xG, rest days)
- [ ] Baseline model (XGBoost)
- [ ] Value bet detection
- [ ] Backtesting framework
- [ ] Web dashboard (Next.js)

## ⚠️ Disclaimer

This project is for educational and research purposes. Gambling carries financial risk — always bet responsibly.
