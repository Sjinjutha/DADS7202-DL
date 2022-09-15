# DADS7202-Deep Learning Hw01

# INTRODUCTION
As a consurance company, managing risk of product launching is a core of business. In this experiment we explored third-party travel insurance data based in Singapore, Using binary classification model with target attribute Claim status(Yes or No). Details shown below will lead you to know more about dataset, methodology and experimentation result.

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

https://www.kaggle.com/datasets/mhdzahier/travel-insurance
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

Reference with normal distribution, 99.7 percent of the data is within three standard deviations (σ) of the mean (μ), we decide to delete data which has age >= 118
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
Overview of score are better.

![Screenshot 2022-09-15 012318](https://user-images.githubusercontent.com/113499057/190234521-22191ee6-04b6-4588-b60f-3ae08bd03aa2.jpg)

we need to provide overfitting model by selected variables from RandomForestClassiflier Model
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

Tuning hyperparameter of RandomForestClassier Model
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
![Screenshot 2022-09-15 145547](https://user-images.githubusercontent.com/113499057/190348161-e58b569b-65b3-4916-ba58-41d01a28e51e.jpg)
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
# List all NVIDIA GPUs as avaialble in this computer (or Colab's session)
!nvidia-smi -L
``````
![Screenshot 2022-09-15 151038](https://user-images.githubusercontent.com/113499057/190356089-3df9265f-cdaa-447d-bb35-dd90c5a36c19.jpg)
``````
import sys
print( f"Python {sys.version}\n" )

## if we have pandas, need to convert data to be numpy
import numpy as np 
print( f"NumPy {np.__version__}\n" )

import matplotlib.pyplot as plt
%matplotlib inline

import tensorflow as tf
print( f"TensorFlow {tf.__version__}" )
print( f"tf.keras.backend.image_data_format() = {tf.keras.backend.image_data_format()}" )

# Count the number of GPUs as detected by tensorflow : tensorflow certainly make sure to see GPUs (tensorflow will run on GPUs)
gpus = tf.config.list_physical_devices('GPU')
print( f"TensorFlow detected { len(gpus) } GPU(s):" )
for i, gpu in enumerate(gpus):
  print( f".... GPU No. {i}: Name = {gpu.name} , Type = {gpu.device_type}" )
``````
![Screenshot 2022-09-15 151056](https://user-images.githubusercontent.com/113499057/190356115-842b2694-5d00-42ba-8768-2e6485cc01ff.jpg)

Set fixed seeding values for reproducability during experiments

Skip this cell if random initialization (with varied results) is needed
``````
np.random.seed(1234)
tf.random.set_seed(5678)
``````
``````
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras

from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import make_scorer, accuracy_score

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier

from math import floor
import statistics
from sklearn.metrics import make_scorer, accuracy_score

from imblearn.under_sampling import InstanceHardnessThreshold

from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, LeakyReLU, Input
!pip install bayesian-optimization
from bayes_opt import BayesianOptimization

#!pip install scikeras
#from scikeras.wrappers import KerasRegressor, KerasClassifier 
#warnings.filterwarnings('ignore')
#pd.set_option("display.max_columns", None)

import warnings
warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", None)
``````
``````
df['Age'] = df['Age'].astype('int')
df = pd.get_dummies(df)

# Re-scale 
col_rescale = ['Duration', 'Net Sales', 'Commision (in value)',  'Age']
rob_scale = RobustScaler() #StandardScaler() 

for i in col_rescale :
    for j in  df.columns :
        if i == j :
            # RobustScaler is less prone to outliers.
            df[i+'_rescaled'] = rob_scale.fit_transform(df[i].values.reshape(-1,1))
            
drop_col = ["Unnamed: 0"] + col_rescale

# Drop and adjust columns
df.drop(drop_col, axis=1, inplace=True)

for i in df.columns[-4:] :
    mv = df.pop(i)
    df.insert(len(df.columns[-4:]), i, mv, allow_duplicates=False)

df.insert(0, "Claim", df.pop("Claim"))

df3 = df

df
``````
![Screenshot 2022-09-15 151319](https://user-images.githubusercontent.com/113499057/190356181-32a32dd1-5b5c-4cc9-892c-f9c2a7133ecf.jpg)

Splitting the Data (Original DataFrame)
``````
df1 = df

not_claim = 100 * df1.Claim.value_counts()[0] / len(df)
claim_ =    100 * df1.Claim.value_counts()[1] / len(df)
print(f"The one who got 'NOT claim' : {round(not_claim, 2)} % of the dataset.")
print(f"The one who got 'claim'     : {round(claim_, 2)} % of the dataset.")
print(f"Old ratio (test:train) : {round((claim_ / not_claim), 2)}\n")

# Set fixed seeding values for reproducability during experiments
# Skip this cell if random initialization (with varied results) is needed
np.random.seed(1234)
tf.random.set_seed(5678)

X_1 = df1.drop('Claim', axis=1)
y_1 = df1['Claim']

sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

for train_index, test_index in sss.split(X_1, y_1):
    print("Train:", train_index, "Test:", test_index)
    original_Xtrain, original_Xtest = X_1.iloc[train_index], X_1.iloc[test_index]
    original_ytrain, original_ytest = y_1.iloc[train_index], y_1.iloc[test_index]
    
print(f"\nNew ratio (test:train) : {round(len(original_Xtest) / len(original_Xtrain), 2)}")
``````
![Screenshot 2022-09-15 151347](https://user-images.githubusercontent.com/113499057/190363922-043cb2dc-b8ca-42ed-b170-48df9cf38c7f.jpg)
``````
# Turn into an array
original_Xtrain = original_Xtrain.values
original_Xtest = original_Xtest.values
original_ytrain = original_ytrain.values
original_ytest = original_ytest.values

# See if both the train and test label distribution are similarly distributed
train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)
test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)
print('-' * 100)
print('Label Distributions: \n')
print(train_counts_label/ len(original_ytrain))
print(test_counts_label/ len(original_ytest))
``````
![Screenshot 2022-09-15 151410](https://user-images.githubusercontent.com/113499057/190363951-b0d3e84e-23cd-4a92-902e-7e0e39055e6a.jpg)

Random Under-Sampling : which basically consists of removing data in order to have a more balanced dataset and thus avoiding our models to overfitting


Assuming we want a 80/20 ratio but select all y (in term of 30% of total)
``````
ratio_want = int((100 / 30) * df.Claim.value_counts()[1]) - int(df.Claim.value_counts()[1])

## shuffle dataset
df_1 = df1.sample(frac=1)

claim_df = df_1.loc[df["Claim"] == 1]
not_claim_df = df_1.loc[df["Claim"] == 0][ : ratio_want]

normal_distributed_df = pd.concat([claim_df, not_claim_df])

# Shuffle dataframe rows
new_df = normal_distributed_df.sample(frac=1, random_state=42)

new_df
``````
![Screenshot 2022-09-15 151438](https://user-images.githubusercontent.com/113499057/190363976-0beaf84b-f97d-4d73-b839-8c2778dc63c9.jpg)
``````
print("Distribution of 'Claim' in the subsample dataset")
print(round(new_df['Claim'].value_counts()/len(new_df),2))

colors = ['red', 'blue']
sns.countplot('Claim', data=new_df, palette=colors)
plt.title('Equally Distributed Claim', fontsize=14)
plt.show()
``````
![Screenshot 2022-09-15 151611](https://user-images.githubusercontent.com/113499057/190362838-a5daee33-2287-4f90-a234-0504d1951f9b.jpg)

Correlation matrices 

Make sure we use the subsample in our correlation

On the other words, we decided to find features that probably making bias on our data (results for "claim") 
by examining corelated variables and eliminating those features

``````
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24,20))

# Entire DataFrame
corr = df1.corr()
sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax1)
ax1.set_title("Imbalanced Correlation Matrix \n (don't use for reference)", fontsize=14)

sub_sample_corr = new_df.corr()
sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax2)
ax2.set_title('SubSample Correlation Matrix \n (use for reference)', fontsize=14)
plt.show()
``````
![Screenshot 2022-09-15 151723](https://user-images.githubusercontent.com/113499057/190362859-7d0edd24-9a3c-4c09-874f-12a75e8be656.jpg)
![Screenshot 2022-09-15 151723](https://user-images.githubusercontent.com/113499057/190362859-7d0edd24-9a3c-4c09-874f-12a75e8be656.jpg)
Eliminate columns which got higher corr > 0.8
``````
list_drop = []

for index,col in list(zip(corr.unstack().index.to_list(), corr.unstack().to_list())) :
    if (index[0] != index[1]) and (col > abs(0.80)):
        print(index, col)
        
        if (index[1] not in list_drop) and (index[0] not in list_drop) :
            list_drop.append(index[1])

list_drop
``````
![Screenshot 2022-09-15 152137](https://user-images.githubusercontent.com/113499057/190372104-72b98609-8e99-4ff4-baeb-12c65cd74ba3.jpg)

Drop columns already
``````
new_df.drop(list_drop, axis=1, inplace=True)
new_df
``````
![Screenshot 2022-09-15 152210](https://user-images.githubusercontent.com/113499057/190372178-cd1ffbbe-4640-4270-a9e1-4e004d2a6e2f.jpg)

Data format: data type : need to be a correct/certain data type

Most DL frameworks use float32 as a default data type
``````
X_train_1 = X_train_1.astype(np.float32)
X_test_1 = X_test_1.astype(np.float32)

print( f"X_train.shape={X_train_1.shape} , X_train.dtype={X_train_1.dtype} , min(X_train)={np.min(X_train_1)} , max(X_train)={np.max(X_train_1)}" )
print( f"X_test.shape={X_test_1.shape} , X_test.dtype={X_test_1.dtype} , min(X_test)={np.min(X_test_1)} , max(X_test)={np.max(X_test_1)}" )
``````
![Screenshot 2022-09-15 152241](https://user-images.githubusercontent.com/113499057/190372269-ff3cef47-a0f5-49bb-8e8a-0301a8990fd5.jpg)
``````
input_dim_num = X_train_1.shape[1] # the number of features per one input
output_dim = 1     # the number of output classes - 1
EPOCHS = 500
BATCH_SIZE = 128
patience_me = 32
learning_rate = 0.5
decay_rate = 5e-6

# Sequential model
model = tf.keras.models.Sequential()

# Input layer
model.add(Input(shape=(input_dim_num,)))
# Hidden layer
model.add(Dense(32, activation='softplus', name='hidden1'))
model.add(BatchNormalization(axis=-1, name='bn1'))
model.add(Dense(64, activation='softplus', name='hidden2'))
model.add(BatchNormalization(axis=-1, name='bn2'))
model.add(Dense(32, activation='softplus', name='hidden3'))
model.add(Dropout(0.3)) # drop rate = 30%
# Output layer
model.add(Dense(output_dim, activation='softplus', name='output'))

# Compile with default values for both optimizer and loss
sgd = SGD(learning_rate=learning_rate, decay=decay_rate)
adam = Adam(learning_rate=learning_rate)

model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['acc']) # sparse_categorical_crossentropy

## OR 
# model.compile( optimizer=tf.keras.optimizers.Adam(learning_rate=0.001) , 
               # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False) , metrics=['acc'] )

model.summary()
``````
![Screenshot 2022-09-15 152423](https://user-images.githubusercontent.com/113499057/190373268-c1805e49-f38a-4f55-bc17-405e27d5e014.jpg)
``````
checkpoint_filepath = "bestmodel_epoch{epoch:02d}_valloss{val_loss:.2f}.hdf5"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint( filepath=checkpoint_filepath,
                                                                save_weights_only=True,
                                                                monitor='val_acc',
                                                                mode='max',
                                                                save_best_only=True)
es_1 = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=patience_me, restore_best_weights=True)
``````
Iteration round 1, 2, 3, 4, 5
``````
number_seed = [1234, 123, 12, 42, 1]
accuracy_list = []

def round_five_iter(number_seed, verbose_see=1) :
    np.random.seed(i)
    tf.random.set_seed(i)
    print(f"This is a round {i+1}")

    model_hist_1 = model.fit(X_train_1, y_train_1, batch_size=128, epochs=EPOCHS, verbose=1, 
                        validation_split=0.3, callbacks=[es_1])
    
    y_pred_1 = model.predict(X_test_1, batch_size=256, verbose=verbose_see, callbacks=[es_1] )
    results_1 = model.evaluate(X_test_1, y_test_1)
    print( f"round {i+1}: {model.metrics_names} = {results_1}" )

    accuracy_list.append(results_1[1])

    # Summarize history for accuracy
    plt.figure(figsize=(15,5))
    plt.plot(model_hist_1.history['acc'])
    plt.plot(model_hist_1.history['val_acc'])
    plt.title('Train accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.grid()
    plt.show()
   

    # Summarize history for loss
    plt.figure(figsize=(15,5))
    plt.plot(model_hist_1.history['loss'])
    plt.plot(model_hist_1.history['val_loss'])
    plt.title('Train loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.grid()
    plt.show()

    print(f"{'-'*100}")
``````

``````
### Round 1
End in 73 epochs 
``````
round_five_iter(1234,1)

![Screenshot 2022-09-15 152828](https://user-images.githubusercontent.com/113499057/190375992-6964e1fc-53b6-4a8a-9620-96e9277cf8cf.jpg)

``````
### Round 2
End in 88 epochs 
``````
round_five_iter(123, 1)

![Screenshot 2022-09-15 152855](https://user-images.githubusercontent.com/113499057/190375971-e67dace0-bf86-404b-a551-5d7f081bb9fe.jpg)
``````
### Round 3
End in 69 epochs 
``````
round_five_iter(12, 0)

![Screenshot 2022-09-15 152931](https://user-images.githubusercontent.com/113499057/190375953-02ee9f39-9ce9-41ae-8f12-38952c1ea6b0.jpg)
``````
### Round 4
End in 57 epochs 
``````
round_five_iter(42, 0)

![Screenshot 2022-09-15 153018](https://user-images.githubusercontent.com/113499057/190375926-8962b4c4-0500-4232-8e27-2548138236df.jpg)
``````
### Round 5
End in 57 epochs 
``````
round_five_iter(1, 0)

![Screenshot 2022-09-15 153127](https://user-images.githubusercontent.com/113499057/190375918-9f90e41c-70c6-4046-9dfe-2dc6f765aa02.jpg)

measure accuracy_list

- training time per epoch : 7ms 

``````
print(f"accuracy_list: {accuracy_list}")

mean_acc = statistics.mean(accuracy_list)
SD_acc = statistics.stdev(accuracy_list)
 
print("Mean is :", mean_acc)
print("SD is :", SD_acc)
``````
![Screenshot 2022-09-15 171740](https://user-images.githubusercontent.com/113499057/190379319-8ee78dc5-909a-4912-baba-60f39408288e.jpg)

## Conclusion
<img width="285" alt="Screenshot 2022-09-15 174717" src="https://user-images.githubusercontent.com/97572167/190385320-27702709-e25e-4c55-a32f-7bc09f2bd1f0.png">

From the results, it was found that MLP gave the highest accuracy = 0.77743±0.0084 and training time = 7ms per epoch.

Note : In general, we will settle model in round 2 to be our model for use-case if we were ruled by accuracy rate.

## Discussion

The main problems that we encountered are
[1] Imbalance dataset between y and x variables : As mentioned above, the proportion of y and x is a huge amount which may mislead assemble our model 
We take some action as you saw.
[2] Unexpectable trial MLP results in fitting step : This problem occured in time of fitting MLP model.
Even though how effort we spent on fitting, the model are steady. Espectially accuaracy rate.
We suspected this tremendous results that probably lead in a fake/false way. 
For example, the giganic accuracy probably gained from predicting almost totally 0 class.(Hardly to predict 1 class in the huge dataset of 0 class)

![Picture1](https://user-images.githubusercontent.com/50355214/190434878-587c7578-174a-4c1b-a392-a68436ca157a.png)

We decided to try in many kind of method such as 
altering "learning rate", number of "nodes, layers, epochs, batch size and dropout_rate" 
However, there are the critical point to fix this problem 3 spots, validation_split ratio, 
activation (change to be "softplus") and optimizer(change to be "sgd")



## Reference
[1] (2022) 'Explaining the Empirical Rule for Normal Distribution' from https://builtin.com/data-science/empirical-rule

[2] (2020) 'Instance Hardness Threshold' from https://towardsdatascience.com/instance-hardness-threshold-an-undersampling-method-to-tackle-imbalanced-classification-problems-6d80f91f0581

[3] (2018) 'Feature Selection Using Random' from forest https://towardsdatascience.com/feature-selection-using-random-forest-26d7b747597f

[4] (2018) 'Hyperparameter Tuning the Random' Forest in Python from https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

[5] (2018) 'Feature Selection Using Random' from forest https://towardsdatascience.com/feature-selection-using-random-forest-26d7b747597f

[6] (2018) 'Hyperparameter Tuning the Random' Forest in Python from https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

[7] 'Classification on imbalanced data' from https://www.tensorflow.org/tutorials/structured_data/imbalanced_data

[8] 'Credit Fraud || Dealing with Imbalanced Datasets' from https://www.kaggle.com/code/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets/notebook

[9] 'Optimizers' from https://keras.io/api/optimizers/

## End Credit
งานชิ้นนี้เป็นส่วนหนึ่งของวิชา DADS7202 Deep Learning หลักสูตรวิทยาศาสตร์มหาบัณฑิต คณะสถิติประยุกต์ สถาบันบัณฑิตพัฒนบริหารศาสตร์

กลุ่ม สู้ DL แต่ DL สู้กลับ 
สมาชิก: (1) 64xxxxxx03 (2) 64xxxxxx06 (3) 64xxxxxx13 (4) 64xxxxxx20
