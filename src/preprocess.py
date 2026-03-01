import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_process(path : str) -> pd.DataFrame:
    data = pd.read_csv(path)
    print(data.info()) # see the count values and dtype for each feature
    print(data.isnull().sum()) # check if there are null values
    print(data['occured'].value_counts(normalize=True)) # check if the target values are balance

    # remove unecessary features
    data = data.drop(['Total Deaths', 'Total Affected', 'Disaster Type', 'time', 'distance'], axis=1)
    
    # feature engineering - think of releveant correlation
    data["rainfall_elevation"] = data["Rainfall"] / (data['Elevation'] + 1)
    data["terrain_risk"] = data["Rainfall"] / (data["Elevation"] - data["Elevation"].min() + 10)
    data["rain_slope"] = data['Rainfall'] * data["Slope"]
    
    for col in ['rainfall_elevation', 'terrain_risk', 'rain_slope']:
        min_val = data[col].min()
        shift = 0
        if min_val <= 0:
            shift = abs(min_val) + 1
        data[col] = np.log1p(data[col] + shift)

    # handle missing values
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(data.median(numeric_only=True), inplace=True)
    
    # visulization for the correlation of each features
    plt.figure(figsize=(15,8))
    sns.heatmap(data.corr(), annot=True, cmap="YlGnBu")
    
    return data






