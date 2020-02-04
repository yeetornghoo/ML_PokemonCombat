import pandas as pd
import numpy as np


def run(data):
    # DROP pokemon_type_2
    data = data.drop(['pokemon_type_2'], axis=1)

    # FILL weight_kg WITH AVERAGE VALUE
    data['weight_kg'] = data['weight_kg'].astype(float)
    data['weight_kg'].fillna((data['weight_kg'].mean()), inplace=True)

    # FILL height_m WITH AVERAGE VALUE
    data['height_m'] = data['height_m'].astype(float)
    data['height_m'].fillna((data['height_m'].mean()), inplace=True)

    return data

#print(data.head())
#print(data.iloc[0:10,2:5].isna())
#column_dt = data["pokemon_type_2"]
#print(column_dt)
#print(data["pokemon_type_2"]==np.nan)