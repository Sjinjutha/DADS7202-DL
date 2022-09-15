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
![Screenshot 2022-09-14 234158](https://user-images.githubusercontent.com/113499057/190225773-0e458d58-4797-4f6a-a4ad-3ce1d82421c1.jpg)

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
![Screenshot 2022-09-14 234158](https://user-images.githubusercontent.com/113499057/190225773-0e458d58-4797-4f6a-a4ad-3ce1d82421c1.jpg)

Check unique element in qualitative variables
``````
column_keys=df.select_dtypes(include=['object']).columns.tolist()
for key in column_keys:
    print('Unique elements of',key,'are: ')
    print(df[key].unique(),end='\n')
    print(end='\n')
``````
![Screenshot 2022-09-14 234238](https://user-images.githubusercontent.com/113499057/190226352-6145c0ee-1ec2-4ed0-a973-245e64e3c599.jpg)
``````
# Agency = Agency Type so delete Agency variable?

A = pd.DataFrame(df.groupby(["Agency","Agency Type", "Claim"]).size(), columns=['Frequency'])
A['Percent'] = round((A['Frequency'] / n)*100 , 2)
A
``````
![Screenshot 2022-09-14 234458](https://user-images.githubusercontent.com/113499057/190226775-95df3c94-4658-48ea-985b-4901782befab.jpg)

Explore target variable because our data is imbalanced data.
``````
yes = df[df.Claim == 'Yes']
n_yes = yes.shape[0]

DN = pd.DataFrame(yes.groupby(["Destination"]).size(), columns=['Frequency'])
DN['Percent'] = round((DN['Frequency'] / n_yes)*100 , 2)
DN.sort_values('Frequency', ascending=False).head(10)
``````
![Screenshot 2022-09-14 234329](https://user-images.githubusercontent.com/113499057/190226623-d260b3b7-46af-490c-bc28-362bf1c5bacf.jpg)

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
data = pd.get_dummies(df, columns = ['Product Name','Destination'])
data
``````
![Screenshot 2022-09-15 001037](https://user-images.githubusercontent.com/113499057/190228595-e843638e-f5ff-4ad9-8f5c-edd04403f65d.jpg)
``````
data.drop('Destination_SINGAPORE', axis=1, inplace=True)
``````
## Mechine Learning
``````
from sklearn.preprocessing import StandardScaler
``````
``````
Y = pd.DataFrame(data.loc[:,'Claim'], columns=['Claim'])

data.drop('Claim', axis=1, inplace=True)
X = data
``````
Since most variables are dummy variables which has value only 0 and 1, so we did data scaling.
``````
scaler = StandardScaler()
scaler.fit(X.loc[:,"Duration":"Age"])
X_scale = scaler.transform(X.loc[:,"Duration":"Age"])
X.loc[:,"Duration":"Age"] = X_scale
X
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
Remember that our data is imbalanced data. Need to show how difference between doing imbalanced or not.
First, we do an traditional data.
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
![Screenshot 2022-09-15 003128](https://user-images.githubusercontent.com/113499057/190229559-ca6de4ac-f0a6-4859-afb2-a4767a0eb366.jpg)
![Screenshot 2022-09-15 003409](https://user-images.githubusercontent.com/113499057/190229586-d3cb33bc-4e4d-48fe-b0e9-23f097ec9daf.jpg)
``````
values.insert(0,['Model','f1_score','roc_auc_score','recall_score','precision_score'])
results = pd.DataFrame(values[1:], columns=values[0])
results
``````
![Screenshot 2022-09-15 003514](https://user-images.githubusercontent.com/113499057/190229948-2f9db5c5-551d-4a4f-ab7a-4e3ddfca9e37.jpg)
Then we do data after imbalanced data.
``````
# Undersampling for imbalanced data
iht = InstanceHardnessThreshold(random_state=42)
X_res, Y_res = iht.fit_resample(X, Y)
``````
``````
claim_res = pd.DataFrame(Y_res.groupby(["Claim"]).size(), columns=['Frequency'])
claim_res['Percent'] = round((claim_res['Frequency'] / Y_res.shape[0])*100 , 2)
claim_res
``````
![Screenshot 2022-09-15 003600](https://user-images.githubusercontent.com/113499057/190230619-e79d4abd-d395-4cbf-ab7c-dc5988a17530.jpg)
``````
X_res_train, X_res_test, y_res_train, y_res_test = train_test_split(X_res, Y_res, test_size=0.2, random_state=42, stratify=Y_res)
``````
``````
values = []
models = [ RandomForestClassifier(), LogisticRegression(), DecisionTreeClassifier(random_state=42), SVC(), KNeighborsClassifier(), XGBClassifier()]
for m in models:
  m.fit(X_res_train, y_res_train)
  y_res_pred = m.predict(X_res_test)
  print(m)
  print(classification_report(y_res_test, y_res_pred)[1])
  print(confusion_matrix(y_res_test, y_res_pred))
  values.append([str(m)[:10], f1_score(y_res_test,y_res_pred), roc_auc_score(y_res_test,y_res_pred), recall_score(y_res_test,y_res_pred), precision_score(y_res_test,y_res_pred)])
  print('='*60)
``````
![Screenshot 2022-09-15 012643](https://user-images.githubusercontent.com/113499057/190233921-edf2036e-d090-4f1f-bce6-fca9ef309d35.jpg)

``````
values.insert(0,['Model','f1_score','roc_auc_score','recall_score','precision_score'])
results = pd.DataFrame(values[1:],columns=values[0])
results
``````
![Screenshot 2022-09-15 012318](https://user-images.githubusercontent.com/113499057/190234521-22191ee6-04b6-4588-b60f-3ae08bd03aa2.jpg)
``````
sel = SelectFromModel(RandomForestClassifier())
sel.fit(X_res_train, y_res_train)

sel.get_support()
``````
![Screenshot 2022-09-15 013425](https://user-images.githubusercontent.com/113499057/190235542-9e1f67c0-2ff4-4ea4-ae80-63b12cb5ed6f.jpg)
``````
selected_feat = X_res_train.columns[(sel.get_support())]
print(selected_feat)
``````
![Screenshot 2022-09-15 013456](https://user-images.githubusercontent.com/113499057/190235628-7520788b-a659-4661-a4fd-48b04eae11ab.jpg)
``````
X_train_selected = X_res_train.loc[:,selected_feat]
X_test_selected = X_res_test.loc[:,selected_feat]
``````
![Screenshot 2022-09-15 013634](https://user-images.githubusercontent.com/113499057/190235647-d1cc4775-84df-46c8-a114-62d02470040a.jpg)
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
![Screenshot 2022-09-15 135457](https://user-images.githubusercontent.com/113499057/190335737-e72dfe12-d5bb-4460-bfa3-1e59822d0bd2.jpg)
``````
rf = RandomForestClassifier(class_weight='balanced', criterion='entropy', max_depth=14, random_state=42)

rf.fit(X_train_selected, y_res_train)
y_selected_pred = rf.predict(X_test_selected)
  
print(classification_report(y_res_test, y_selected_pred)[1])
print(confusion_matrix(y_res_test, y_selected_pred))

values = []
values.append([str(rf)[:10], f1_score(y_res_test,y_selected_pred), roc_auc_score(y_res_test,y_selected_pred), 
               recall_score(y_res_test,y_selected_pred), precision_score(y_res_test,y_selected_pred)])
``````
![Screenshot 2022-09-15 135536](https://user-images.githubusercontent.com/113499057/190335766-2ab8fac0-260a-48f6-b470-0ca0cb56fdcf.jpg)
``````
values.insert(0,['Model','f1_score','roc_auc_score','recall_score','precision_score'])
results = pd.DataFrame(values[1:], columns=values[0])
results
``````
![Screenshot 2022-09-15 135553](https://user-images.githubusercontent.com/113499057/190335786-365cb191-26a2-4b71-afda-e7a75f691d1f.jpg)

## Deep Learning (MLP)
``````

``````
``````

``````
``````

``````
``````

``````
``````

``````
``````

``````
``````

``````
``````

``````
## Reference
[1] (2020) 'Instance Hardness Threshold' from https://towardsdatascience.com/instance-hardness-threshold-an-undersampling-method-to-tackle-imbalanced-classification-problems-6d80f91f0581

[2] (2018) 'Feature Selection Using Random' from forest https://towardsdatascience.com/feature-selection-using-random-forest-26d7b747597f

[3] (2018) 'Hyperparameter Tuning the Random' Forest in Python from https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

## End Credit
งานชิ้นนี้เป็นส่วนหนึ่งของวิชา DADS7202 Deep Learning หลักสูตรวิทยาศาสตร์มหาบัณฑิต คณะสถิติประยุกต์ สถาบันบัณฑิตพัฒนบริหารศาสตร์

กลุ่ม สู้ DL แต่ DL สู้กลับ สมาชิก: (1) 64xxxxxx03 (2) 64xxxxxx06 (3) 64xxxxxx13 (4) 64xxxxxx20
