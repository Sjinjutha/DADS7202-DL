# DADS7202-Deep Learning Hw01

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
Check data information & missing value
``````
df.info()
``````
``````
n = df.shape[0]
print(f"Number of rows {n}")
``````
``````
# percent of missing "Gender" 
print('Percent of missing "Gender" records is %.2f%%' %((df['Gender'].isnull().sum()/df.shape[0])*100))
``````
Percent of missing "Gender" records is 71.23%, so we decide to drop Gender variable.
``````
df = df.drop(['Gender'], axis = 1)
df.head()
``````
``````
new_cols = ['Agency','Agency Type','Distribution Channel','Product Name','Duration','Destination','Net Sales','Commision (in value)','Age','Claim']
df = df.reindex(columns=new_cols)
df.head()
``````
Check number of data per binary classification (Cliam: Yes,No)
``````
claim = pd.DataFrame(df.groupby(["Claim"]).size(), columns=['Frequency'])
claim['Percent'] = round((claim['Frequency'] / n)*100 , 2)
claim
``````
Even though Age variable does not has missing value but it has outlier.
``````
figure_age = plt.figure(figsize = (10, 5))
plt.hist(df['Age'])
plt.xlabel("Age")
plt.ylabel("Number of people")
plt.title("Distribution of Age")
plt.show()
``````
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
``````
n = df.shape[0]
n
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
df.head(10)
``````
``````
dm = pd.get_dummies(df, columns = ['Product Name','Destination'])
dm
``````
``````



