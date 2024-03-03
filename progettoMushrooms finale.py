import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# https://www.kaggle.com/datasets/ulrikthygepedersen/mushroom-attributes?rvi=1
df = pd.read_csv("mushroom.csv")

df.head()
df.info()
df.describe()
print("Dataset shape: ", df.shape)

df['class'].unique()
df['class'].value_counts()


count = df['class'].value_counts()

#visualizziamo il conteggio dei funghi velenosi e commestibili con Matplotlib e Pyplot
plt.figure(figsize=(8,7))
plt.ylabel('Count', fontsize=12)
plt.xlabel('Class', fontsize=12)
plt.title('Number of poisonous/edible mushrooms')
plt.savefig("mushrooms1.png", format='png', dpi=500)
plt.show()

#con LabelEncoder convertiamo i dati da categoriali a ordinali
df = df.astype('category')
labelencoder=LabelEncoder()
for column in df.columns:
    df[column] = labelencoder.fit_transform(df[column])
df.head()


#rimuoviamo la colonna veil-type perché avrà tutti valori uguali a 0 e dunque non influisce sui dati
df = df.drop(["veil-type"],axis=1)

df_div = pd.melt(df, "class", var_name="Characteristics")
fig, ax = plt.subplots(figsize=(16,6))

p = sns.violinplot(ax = ax, x="Characteristics", y="value", hue="class", split = True, data=df_div, inner = 'quartile', palette = 'Set1')

df_no_class = df.drop(["class"],axis = 1)
p.set_xticklabels(rotation = 90, labels = list(df_no_class.columns))

#osserviamo la correlazione tra variabili
plt.figure(figsize=(14,12))
sns.heatmap(df.corr(),linewidths=.1,cmap="Purples", annot=True, annot_kws={"size": 7})
plt.yticks(rotation=0)
plt.savefig("corr.png", format='png', dpi=400, bbox_inches='tight')

#osserviamo la variabile gill-color, essendo la meno correlata può essere la più importante per la classificazione
df[['class', 'gill-color']].groupby(['gill-color'], as_index=False).mean().sort_values(by='class', ascending=False)

#preparazione dati e albero decisionale
X = df.drop(['class'], axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)
y_pred_probs_dt = dt.predict_proba(X_test)
classification_rep_dt = classification_report(y_test, y_pred_dt)

print('Decision Tree - Classification Report:\n', classification_rep_dt)
print('Decision Tree - Cross Entropy:', log_loss(y_test, y_pred_probs_dt))

#test accuracy
print("Test Accuracy: {}%".format(round(dt.score(X_test, y_test)*100, 2)))


features_list = X.columns.values
feature_importance = dt.feature_importances_
sorted_idx = np.argsort(feature_importance)
plt.figure(figsize=(8,7))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center', color ="red")
plt.yticks(range(len(sorted_idx)), features_list[sorted_idx])
plt.xlabel('Importance')
plt.title('Feature importance')
plt.draw()
plt.savefig("featureimp.png", format='png', dpi=500, bbox_inches='tight')
plt.show()


#MATRICE DI CONFUSIONE PER IL CLASSIFICATORE DELL'ALBERO DECISIONALE
cm_dt = confusion_matrix(y_test, y_pred_dt)
x_axis_labels_dt = ["Edible", "Poisonous"]
y_axis_labels_dt = ["Edible", "Poisonous"]
f_dt, ax_dt = plt.subplots(figsize =(7,7))
plt.xlabel("PREDICTED LABEL")
plt.ylabel("TRUE LABEL")
plt.title('Confusion Matrix for Decision Tree Classifier')
plt.savefig("dtcm.png", format='png', dpi=500, bbox_inches='tight')
plt.show()

#CLASSIFICAZIONE CASUALE DELLE FORESTE (RANDOM FOREST)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print("Test Accuracy: {}%".format(round(rf.score(X_test, y_test)*100, 2)))

#RAPPORTO DI CLASSIFICAZIONE DEL RANODM FOREST CLASSIFIER

y_pred_rf = rf.predict(X_test)
print("Random Forest Classifier report: \n\n", classification_report(y_test, y_pred_rf))

#MATRICE DI CONFUSIONE PER CLASSIFICATORE DI FORESTA CASUALE

cm_rf = confusion_matrix(y_test, y_pred_rf)
x_axis_labels_rf = ["Edible", "Poisonous"]
y_axis_labels_rf = ["Edible", "Poisonous"]
f, ax = plt.subplots(figsize =(7,7))
sns.heatmap(cm_rf, annot = True, linewidths=0.2, linecolor="black", fmt = ".0f", ax_rf=ax, cmap="Purples", xticklabels= x_axis_labels_rf, yticklabels= y_axis_labels_rf)
plt.xlabel("PREDICTED LABEL")
plt.ylabel("TRUE LABEL")
plt.title('Confusion Matrix for Random Forest Classifier')
plt.savefig("rfcm.png", format='png', dpi=500, bbox_inches='tight')
plt.show()

#PREDIZIONI
preds = dt.predict(X_test)
print(preds[:36])
print(y_test[:36].values)







