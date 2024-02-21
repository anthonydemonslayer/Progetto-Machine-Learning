import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


from ucimlrepo import fetch_ucirepo
from datasets import load_dataset

dataset = load_dataset("jlh/uci-mushrooms")

# fetch dataset
mushroom = fetch_ucirepo(id=73)

# data (as pandas dataframes)
X = mushroom.data.features
y = mushroom.data.targets

# metadata
print(mushroom.metadata)

# variable information
print(mushroom.variables)

# https://www.kaggle.com/datasets/ulrikthygepedersen/mushroom-attributes?rvi=1
df = pd.read_csv('mushroom.csv')

df = df.drop(columns=['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing',
                      'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
                      'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type',
                      'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat', 'class'],
             inplace=true)

df.head()
df.info()
df.describe()
print("Dataset shape: ", df.shape)
