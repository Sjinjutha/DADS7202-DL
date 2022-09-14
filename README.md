# DADS7202-Deep Learning Hw01

# Data Preparing
```
import pandas as np
import numpy as np
import matplotlib.pyplot as plt
```
```
df = pd.read_csv("/content/drive/MyDrive/travel_insurance.csv")
df.head()
``````
``````
df.info()
``````
``````
n = df.shape[0]
n
``````
``````
df.isnull().sum()
``````
``````
df = df.drop(['Gender'], axis = 1)
df.head()
``````
``````
new_cols = ['Agency','Agency Type','Distribution Channel','Product Name','Duration','Destination','Net Sales','Commision (in value)','Age','Claim']
df = df.reindex(columns=new_cols)
df.head()
``````
``````
claim = pd.DataFrame(df.groupby(["Claim"]).size(), columns=['Frequency'])
claim['Percent'] = round((claim['Frequency'] / n)*100 , 2)
claim
``````
``````
figure_age = plt.figure(figsize = (10, 5))
plt.hist(df['Age'])
plt.xlabel("Age")
plt.ylabel("Number of people")
plt.title("Distribution of Age")
plt.show()
``````
``````
df['Age'] = np.where(df['Age'] >= df['Age'].quantile(0.997), np.nan, df['Age'])
df = df.dropna()
``````
``````
figure_age = plt.figure(figsize = (10, 5))
plt.hist(df['Age'])
plt.xlabel("Age")
plt.ylabel("Number of people")
plt.title("Distribution of Age")
plt.show()
``````
``````
n = df.shape[0]
n
``````
``````
column_keys=df.select_dtypes(include=['object']).columns.tolist()
for key in column_keys:
    print('Unique elements of',key,'are: ')
    print(df[key].unique(),end='\n')
    print(end='\n')
``````
``````
df_dummy = pd.get_dummies(df, columns = ['Agency','Agency Type','Distribution Channel','Product Name','Destination'])
df_dummy
``````
``````
yes = df[df.Claim == 'Yes']
n_yes = yes.shape[0]
``````
``````
DN = pd.DataFrame(yes.groupby(["Destination"]).size(), columns=['Frequency'])
DN['Percent'] = round((DN['Frequency'] / n_yes)*100 , 2)
DN.sort_values('Frequency', ascending=False).head(10)
``````
``````
# Agency Type : Airlines=1, TravelAgency:0
df["Agency Type"] = np.where((df["Agency Type"] == 'Airlines'), 1,0)

# Distribution Channel : Online=1, Offline=0
df["Distribution Channel"] = np.where((df["Distribution Channel"] == 'Online'), 1,0)

# Destination : Singapore=1, others:0
df["Des_Singapore"] = np.where((df["Destination"] == 'SINGAPORE'), 1,0)

# Target variable 'Claim' : Yes=1, No=0
df["Claim"] = np.where((df["Claim"] == 'Yes'), 1,0)
df.head(10)
``````
``````
dm = pd.get_dummies(df, columns = ['Product Name','Destination'])
dm
``````
``````



