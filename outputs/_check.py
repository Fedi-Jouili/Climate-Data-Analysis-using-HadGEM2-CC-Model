import pandas as pd
from pathlib import Path
d = Path(__file__).parent
df = pd.read_csv(d / 'trend_slope_maps.csv')
print("slope vars:", list(df['variable'].unique()))
df2 = pd.read_csv(d / 'spatial_mean_maps.csv')
print("map vars:", list(df2['variable'].unique()))
df3 = pd.read_csv(d / 'decadal_changes.csv', index_col=0)
print("decadal cols:", list(df3.columns))
print("decadal index:", list(df3.index))
