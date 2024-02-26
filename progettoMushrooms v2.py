import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import osimport graphviz
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

df.head();
df.info();
df.describe();
print("Dataset shape: ", df.shape);

df['class'].unique();
df['class'].value_counts();


count = df['class'].value_counts();

#visualizziamo il conteggio dei funghi velenosi e commestibili con Matplotlib e Pyplot
plt.figure(figsize=(8,7));
plt.ylabel('Count', fontsize=12);
plt.xlabel('Class', fontsize=12);
plt.title('Number of poisonous/edible mushrooms');
plt.savefig("mushrooms1.png", format='png', dpi=500);
plt.show();

#con LabelEncoder convertiamo i dati da categoriali a ordinali
df = df.astype('category');
df.dtypes;
labelencoder=LabelEncoder()
for columns in df.columns {
    df[column] = labelencoder.fit_transform(df[column]);
}
df.head();


#rimuoviamo la colonna veil-type perché avrà tutti valori uguali a 0 e dunque non influisce sui dati
df = df.drop(["veil-type"],axis=1)

df_div = pd.melt(df, “class”, var_name=”Characteristics”);
fig, ax = plt.subplots(figsize=(16,6));

p = sns.violinplot(ax = ax, x=”Characteristics”, y=”value”, hue=”class”, split = True, data=df_div,
inner = ‘quartile’, palette = ‘Set1’);

df_no_class = df.drop([“class”],axis = 1);
p.set_xticklabels(rotation = 90, labels = list(df_no_class.columns));

#osserviamo la correlazione tra variabili
plt.figure(figsize=(14,12));
sns.heatmap(df.corr(),linewidths=.1,cmap="Purples", annot=True, annot_kws={"size": 7});
plt.yticks(rotation=0);
plt.savefig("corr.png", format='png', dpi=400, bbox_inches='tight');

#osserviamo la variabile gill-color, essendo la meno correlata può essere la più importante per la classificazione
df[['class', 'gill-color']].groupby(['gill-color'], as_index=False).mean().sort_values(by='class', ascending=False)

new_var = df[['class', 'gill-color']];
new_var = new_var[new_var['gill-color']<=3.5];
sns.factorplot('class', col='gill-color', data=new_var, kind='count', size=4.5, aspect=.8, col_wrap=4);
#plt.savefig("gillcolor1.png", format='png', dpi=500, bbox_inches='tight')

new_var=df[['class', 'gill-color']];
new_var=new_var[new_var['gill-color']>3.5];
sns.factorplot('class', col='gill-color', data=new_var, kind='count', size=4.5, aspect=.8, col_wrap=4);
#plt.savefig("gillcolor2.png", format='png', dpi=400, bbox_inches='tight')


#preparazione dati e albero decisionale
X = df.drop([‘class’], axis=1);
y = df[“class”]X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1);

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

