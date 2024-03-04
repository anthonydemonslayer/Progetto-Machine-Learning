import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
import time

# https://www.kaggle.com/datasets/ulrikthygepedersen/mushroom-attributes?rvi=1
df = pd.read_csv("mushroom.csv")

df.head()
df.info()
df.describe()
print("Dataset shape: ", df.shape)

df['class'].unique()

# con LabelEncoder convertiamo i dati da categoriali a ordinali
df = df.astype('category')
labelencoder = LabelEncoder()
for column in df.columns:
    df[column] = labelencoder.fit_transform(df[column])
df.head()


# rimuoviamo la colonna veil-type perché avrà tutti valori uguali a 0 e dunque non influisce sui dati
df = df.drop(["veil-type"], axis=1)

df_div = pd.melt(df, "class", var_name="Characteristics")
fig, ax = plt.subplots(figsize=(16, 6))

p = sns.violinplot(ax = ax, x="Characteristics", y="value", hue="class", split = True, data=df_div, inner = 'quartile', palette = 'Set1')

df_no_class = df.drop(["class"], axis = 1)
p.set_xticklabels(rotation = 90, labels = list(df_no_class.columns))

# osserviamo la correlazione tra variabili
plt.figure(figsize=(14, 12))
sns.heatmap(df.corr(), linewidths=.1, cmap="Purples", annot=True, annot_kws={"size": 7})
plt.yticks(rotation=0)
# plt.savefig("corr.png", format='png', dpi=400, bbox_inches='tight')

# osserviamo la variabile gill-color, essendo la meno correlata può essere la più importante per la classificazione
df[['class', 'gill-color']].groupby(['gill-color'], as_index=False).mean().sort_values(by='class', ascending=False)

# preparazione dati e albero decisionale
X = df.drop(['class'], axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)

start_time_dt = time.time()
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)
y_pred_probs_dt = dt.predict_proba(X_test)
end_time_dt = time.time()

classification_rep_dt = classification_report(y_test, y_pred_dt)

print('Decision Tree - Classification Report:\n', classification_rep_dt)
print('Decision Tree - Cross Entropy:', log_loss(y_test, y_pred_probs_dt))
print('Decision Tree - Training Time:', end_time_dt - start_time_dt, 'seconds\n')

# test accuracy
print("Test Accuracy: {}%".format(round(dt.score(X_test, y_test)*100, 2)))


features_list = X.columns.values
feature_importance = dt.feature_importances_
sorted_idx = np.argsort(feature_importance)
plt.figure(figsize=(8, 7))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center', color ="red")
plt.yticks(range(len(sorted_idx)), features_list[sorted_idx])
plt.xlabel('Importance')
plt.title('Feature importance')
plt.draw()
# plt.savefig("featureimp.png", format='png', dpi=500, bbox_inches='tight')
plt.show()


# CLASSIFICAZIONE KNN
start_time_knn = time.time()
best_Kvalue = 0
best_score = 0
for i in range(1, 10):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    if knn.score(X_test, y_test) > best_score:
        best_score = knn.score(X_train, y_train)
        best_Kvalue = i

print("Best KNN Value: {}".format(best_Kvalue))
print("Test Accuracy: {}%".format(round(best_score*100, 2)))

# RAPPORTO DI CLASSIFICAZIONE
y_pred_knn = knn.predict(X_test)
end_time_knn = time.time()

classification_rep_knn = classification_report(y_test, y_pred_knn)
print('KNN - Classification Report:\n', classification_rep_knn)
print('KNN - Cross Entropy:', log_loss(y_test, y_pred_knn))
print('KNN - Training Time:', end_time_knn - start_time_knn, 'seconds\n')

# CLASSIFICAZIONE CASUALE DELLE FORESTE (RANDOM FOREST)

start_time_rf = time.time()
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print("Test Accuracy: {}%".format(round(rf.score(X_test, y_test)*100, 2)))

# RAPPORTO DI CLASSIFICAZIONE DEL RANODM FOREST CLASSIFIER

y_pred_rf = rf.predict(X_test)
end_time_rf = time.time()

classification_rep_rf = classification_report(y_test, y_pred_rf)
print('Random Forest - Classification Report:\n', classification_rep_rf)
print('Random Forest - Cross Entropy:', log_loss(y_test, y_pred_rf))
print('Random Forest - Training Time:', end_time_rf - start_time_rf, 'seconds\n')
# PREDIZIONI
preds = dt.predict(X_test)
print(preds[:36])
print(y_test[:36].values)