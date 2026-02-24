import pandas as pd

def load_league_data(league: str, season: str) -> pd.DataFrame:
    """
    league: 'E0' = Premier League, 'D1' = Bundesliga, 'SP1' = La Liga
    season: '2324' = 2023/24, '2223' = 2022/23
    """
    url = f"https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
    df = pd.read_csv(url)
    return df

# Load last 3 Bundesliga seasons
seasons = ['2324', '2223', '2122']
frames = [load_league_data('D1', s) for s in seasons]
bundesliga = pd.concat(frames, ignore_index=True)

print(bundesliga.shape)
print(bundesliga.columns.tolist( ))