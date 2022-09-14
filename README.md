# DADS7202-Deep Learning Hw01

## About Dataset
A third-party travel insurance servicing company that is based in Singapore.

The attributes:

1. Target: Claim Status (Claim.Status)
2. Name of agency (Agency)
3. Type of travel insurance agencies (Agency.Type)
4. Distribution channel of travel insurance agencies (Distribution.Channel)
5. Name of the travel insurance products (Product.Name)
6. Duration of travel (Duration)
7. Destination of travel (Destination)
8. Amount of sales of travel insurance policies (Net.Sales)
9. Commission received for travel insurance agency (Commission)
10. Gender of insured (Gender)
11. Age of insured (Age)

## Data Preparing

Import library
```
import pandas as np
import numpy as np
import matplotlib.pyplot as plt
```
Download dataset to your google drive and mount it.
```
from google.colab import drive
drive.mount('/content/drive')
df = pd.read_csv("/content/drive/MyDrive/travel_insurance.csv")
df.head()
``````
![messageImage_1663155344482](https://user-images.githubusercontent.com/113499057/190196116-fd964dcc-7588-4966-9010-068cbc39cebe.jpg)
Check data information & missing value
``````
df.info()
``````
![Screenshot 2022-09-14 212635](https://user-images.githubusercontent.com/113499057/190195652-d0b5cb8b-ce40-4894-8cd9-5fec4b89edd9.jpg)
``````
n = df.shape[0]
m = df.shape[1]
print(f"Data contains {n} rows and {m} columns.")
``````
![Screenshot 2022-09-14 221430](https://user-images.githubusercontent.com/113499057/190198230-0080316e-d02b-4bdd-8189-22811f45786b.jpg)
``````
# percent of missing "Gender" 
print('Percent of missing "Gender" records is %.2f%%' %((df['Gender'].isnull().sum()/n)*100))
``````
![Screenshot 2022-09-14 215708](https://user-images.githubusercontent.com/113499057/190197245-cd2c984d-8c7b-4634-b6e6-652e1273f797.jpg)

Percent of missing "Gender" records is 71.23%, so we decide to drop Gender variable.
``````
df = df.drop(['Gender'], axis = 1)
df.head()
``````
![Screenshot 2022-09-14 221513](https://user-images.githubusercontent.com/113499057/190210803-f161ebc2-54a9-49a2-bdc6-65a42d37eca9.jpg)
``````
new_cols = ['Agency','Agency Type','Distribution Channel','Product Name','Duration','Destination','Net Sales','Commision (in value)','Age','Claim']
df = df.reindex(columns=new_cols)
df.head()
``````
![Screenshot 2022-09-14 221530](https://user-images.githubusercontent.com/113499057/190211071-65a4cde0-b810-441f-a6d8-83e1970d0dff.jpg)
Check number of data per binary classification (Cliam: Yes,No). There is imbalanced dataset.
``````
claim = pd.DataFrame(df.groupby(["Claim"]).size(), columns=['Frequency'])
claim['Percent'] = round((claim['Frequency'] / n)*100 , 2)
claim
``````
![Screenshot 2022-09-14 221557](https://user-images.githubusercontent.com/113499057/190211624-cb257fbc-e3a5-43e1-b5d0-debe8e9f703f.jpg)

Even though Age variable does not has missing value but it has outlier.
``````
figure_age = plt.figure(figsize = (10, 5))
plt.hist(df['Age'])
plt.xlabel("Age")
plt.ylabel("Number of people")
plt.title("Distribution of Age")
plt.show()
``````
![Screenshot 2022-09-14 221622](https://user-images.githubusercontent.com/113499057/190212469-5d8fd3ba-56b5-4f22-8165-1fa07c5824ab.jpg)

Reference with normal distribution, we decide to delete data which has age >= 118
``````
df['Age'] = np.where(df['Age'] >= df['Age'].quantile(0.997), np.nan, df['Age'])
df = df.dropna()

figure_age = plt.figure(figsize = (10, 5))
plt.hist(df['Age'])
plt.xlabel("Age")
plt.ylabel("Number of people")
plt.title("Distribution of Age")
plt.show()
``````
![Screenshot 2022-09-14 221644](https://user-images.githubusercontent.com/113499057/190212662-a02c6f46-282c-4c4c-9624-a75edd26f941.jpg)
``````
print(f"After dropped some data : {n} rows and {m} columns.")
``````
Check unique element in qualitative variables
``````
column_keys=df.select_dtypes(include=['object']).columns.tolist()
for key in column_keys:
    print('Unique elements of',key,'are: ')
    print(df[key].unique(),end='\n')
    print(end='\n')
``````
Explore target variable because our data is imbalanced data.
``````
yes = df[df.Claim == 'Yes']
n_yes = yes.shape[0]

DN = pd.DataFrame(yes.groupby(["Destination"]).size(), columns=['Frequency'])
DN['Percent'] = round((DN['Frequency'] / n_yes)*100 , 2)
DN.sort_values('Frequency', ascending=False).head(10)
``````
``````
# Agency = Agency Type so delete Agency variable?

A = pd.DataFrame(df.groupby(["Agency","Agency Type", "Claim"]).size(), columns=['Frequency'])
A['Percent'] = round((A['Frequency'] / n)*100 , 2)
A

``````
Change data from nominal to ratio and add new variable because we see that data which precent of destination which is Singapore is 61.29%.
``````
# Agency Type : Airlines=1, TravelAgency:0
df["Agency Type"] = np.where((df["Agency Type"] == 'Airlines'), 1,0)

# Distribution Channel : Online=1, Offline=0
df["Distribution Channel"] = np.where((df["Distribution Channel"] == 'Online'), 1,0)

# Destination : Singapore=1, others:0
df["Des_Singapore"] = np.where((df["Destination"] == 'SINGAPORE'), 1,0)

# Target variable 'Claim' : Yes=1, No=0
df["Claim"] = np.where((df["Claim"] == 'Yes'), 1,0)

# Create dummy variables only Product Name and Destination variable
dm = pd.get_dummies(df, columns = ['Product Name','Destination'])
dm
``````
## Mechine Learning
``````
from sklearn.preprocessing import StandardScaler
``````
``````
scaler = StandardScaler()
scaler.fit(X.loc[:,"Duration":"Age"])
X_scale = scaler.transform(X.loc[:,"Duration":"Age"])
X.loc[:,"Duration":"Age"] = X_scale
``````
``````
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost.sklearn import XGBClassifier
from imblearn.under_sampling import InstanceHardnessThreshold

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score, confusion_matrix, classification_report

from sklearn.feature_selection import SelectFromModel
``````
``````
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
``````
``````
values = []
models = [ RandomForestClassifier(), LogisticRegression(), DecisionTreeClassifier(), SVC(), KNeighborsClassifier(), XGBClassifier()]
for m in models:
  m.fit(X_train, y_train)
  y_pred = m.predict(X_test)
  print(m)
  print(classification_report(y_test,y_pred)[1])
  print(confusion_matrix(y_test,y_pred))
  values.append([str(m)[:10], f1_score(y_test,y_pred), roc_auc_score(y_test,y_pred), recall_score(y_test,y_pred), precision_score(y_test,y_pred)])
  print('==========================================================')
``````
``````
values.insert(0,['Model','f1_score','roc_auc_score','recall_score','precision_score'])
results = pd.DataFrame(values[1:], columns=values[0])
results
``````
``````
iht = InstanceHardnessThreshold(random_state=42)
X_res, Y_res = iht.fit_resample(X, Y)
``````
``````
claim_res = pd.DataFrame(Y_res.groupby(["Claim"]).size(), columns=['Frequency'])
claim_res['Percent'] = round((claim_res['Frequency'] / Y_res.shape[0])*100 , 2)
claim_res
``````
``````
values.insert(0,['Model','f1_score','roc_auc_score','recall_score','precision_score'])
results = pd.DataFrame(values[1:],columns=values[0])
results
``````
``````
sel = SelectFromModel(RandomForestClassifier())
sel.fit(X_res_train, y_res_train)

sel.get_support()
``````
``````
X_train_selected = X_res_train.loc[:,selected_feat]
X_test_selected = X_res_test.loc[:,selected_feat]
``````
``````
# use GridSerchCV to find the best parameter by exhaustive search over specified parameter values for an estimator.

from sklearn.model_selection import GridSearchCV

clf = RandomForestClassifier(random_state=42)

# defining parameter range
param_grid = {'criterion':['gini','entropy','log_loss'],
              'max_depth':[int(x) for x in range(2,21,1)],
              'max_features':['sqrt','log2'],
              'class_weight':['balanced','balanced_subsample',None]
              }

grid = GridSearchCV(clf, param_grid, cv=10, scoring='f1', return_train_score=True, verbose=1)
  
# fitting the model for grid search
grid_search = grid.fit(X_train_selected, y_res_train)

print(grid_search.best_params_)
``````
``````
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_selected, y_res_train)
y_selected_pred = rf.predict(X_test_selected)
  
print(classification_report(y_res_test, y_selected_pred)[1])
print(confusion_matrix(y_res_test, y_selected_pred))

values = []
values.append([str(rf)[:10], f1_score(y_res_test,y_selected_pred), roc_auc_score(y_res_test,y_selected_pred), 
               recall_score(y_res_test,y_selected_pred), precision_score(y_res_test,y_selected_pred)])
``````
``````
values.insert(0,['Model','f1_score','roc_auc_score','recall_score','precision_score'])
results = pd.DataFrame(values[1:], columns=values[0])
results
``````

## Deep Learning (MLP)

