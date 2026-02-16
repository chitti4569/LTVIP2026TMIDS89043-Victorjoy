
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea

dataset = pd.read_excel('data/flood dataset.xlsx')
print(dataset)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.histplot(dataset['Temp'], kde=True)
plt.show()

sns.boxplot(x=dataset['Temp'])
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
fig = plt.gcf()
fig.set_size_inches(15, 15)
fig = sns.heatmap(
    dataset.corr(),
    annot=True,
    cmap='summer',
    linewidths=1,
    linecolor='k',
    square=True,
    mask=False,
    vmin=-1,
    vmax=1,
    cbar_kws={"orientation": "vertical"},
    cbar=True
)
plt.show()

dataset.head()

print(dataset.info())

print(dataset.describe().T)

print(dataset.isnull().any())

X = dataset.iloc[:, 2:7].values
y = dataset.iloc[:, 9].values

print(dataset.columns)

from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(dataset, test_size=0.25, random_state=10)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from joblib import dump
dump(sc, "transform.save")

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris
data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
dtree = tree.DecisionTreeClassifier()
dtree.fit(X_train, y_train)
dt_pred = dtree.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))
print("Decision Tree Confusion Matrix:\n", confusion_matrix(y_test, dt_pred))
print("Decision Tree Classification Report:\n", classification_report(y_test, dt_pred))

from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris
data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
rf = ensemble.RandomForestClassifier()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))
print("Random Forest Classification Report:\n", classification_report(y_test, rf_pred))

from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris
data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
knn = neighbors.KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
print("KNN Accuracy:", accuracy_score(y_test, knn_pred))
print("KNN Confusion Matrix:\n", confusion_matrix(y_test, knn_pred))
print("KNN Classification Report:\n", classification_report(y_test, knn_pred))

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris
data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
xgb_model = xgb.XGBClassifier(eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
print("XGBoost Accuracy:", accuracy_score(y_test, xgb_pred))
print("XGBoost Confusion Matrix:\n", confusion_matrix(y_test, xgb_pred))
print("XGBoost Classification Report:\n", classification_report(y_test, xgb_pred))

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.datasets import load_iris
data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model1 = LogisticRegression(max_iter=200)
model1.fit(X_train, y_train)
p1 = model1.predict(X_test)
model2 = DecisionTreeClassifier()
model2.fit(X_train, y_train)
p2 = model2.predict(X_test)
model3 = RandomForestClassifier()
model3.fit(X_train, y_train)
p3 = model3.predict(X_test)
model4 = SVC()
model4.fit(X_train, y_train)
p4 = model4.predict(X_test)
print("Model 1 Accuracy:", metrics.accuracy_score(y_test, p1))
print("Model 2 Accuracy:", metrics.accuracy_score(y_test, p2))
print("Model 3 Accuracy:", metrics.accuracy_score(y_test, p3))
print("Model 4 Accuracy:", metrics.accuracy_score(y_test, p4))

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
data = load_breast_cancer()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=29, random_state=2
)
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)
p4 = model.predict(X_test)
print(metrics.confusion_matrix(y_test, p4))
print(metrics.accuracy_score(y_test, p4))
print(metrics.precision_score(y_test, p4))
print(metrics.recall_score(y_test, p4))

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
df = pd.read_excel('data/flood dataset.xlsx')
if df["flood"].dtype == object:
    df["flood"] = df["flood"].map({"Yes": 1, "No": 0})
print("Flood value counts:")
print(df["flood"].value_counts())
X = df[[
    "Cloud Cover",
    "ANNUAL",
    "Jan-Feb",
    "Mar-May",
    "Jun-Sep"
]]
y = df["flood"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=15,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
joblib.dump(model, "floods.save")
print("\nModel retrained successfully!")
