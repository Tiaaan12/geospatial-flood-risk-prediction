import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("data/raw/flood_dataset_classification.csv")
print(data.info()) # See the count values and dtype for each feature
print(data.isnull().sum()) # check if there are null values
print(data['occured'].value_counts)




