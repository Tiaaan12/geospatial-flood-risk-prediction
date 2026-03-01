import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("data/raw/flood_dataset_classification.csv")
print(data.info()) # see the count values and dtype for each feature
print(data.isnull().sum()) # check if there are null values
print(data['occured'].value_counts(normalize=True)) # check if the target values are balance

# remove unecessary features
data = data.drop(['Total Deaths', 'Total Affected', 'Disaster Type', 'time', 'distance'], axis=1)

data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.fillna(data.median(numeric_only=True), inplace=True)






