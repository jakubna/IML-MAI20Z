from scipy.io import arff
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder

# load dataset
f = 'C://Users//nadzi//Desktop//datasets//datasets//cmc.arff'
data, meta = arff.loadarff(f)
dataset = np.array(data.tolist(), dtype=object)

column_names = meta.names() # list containing column names
column_types = meta.types() # list containing column types
column_types_dict = dict(zip(column_names[:-1], column_types[:-1])) # dictionary containing column names and types
# create a initial pandas dataframe
df = pd.DataFrame(data=dataset, columns=column_names)

# detect and replace missing values
if df.isnull().any().sum() > 0:
    for x in meta[:-1]:
        if column_types_dict[x] == 'nominal':
            top_frequent_value = df[x].describe()['top']
            df[x].fillna(top_frequent_value, inplace=True)
        else:
            median = df[x].median()
            df[x].fillna(median, inplace=True)

# create instance of labelencoder
labelencoder = LabelEncoder()
# transform data
for k,v in column_types_dict.items():
    if v == 'nominal':
        labelencoder.fit(sorted(df[k].unique()))
        df[k] = labelencoder.transform(df[k])

# split-out dataset
X = df.iloc[:,:-1]
Y = df.iloc[:, -1]

# fit MinMax scaler on data
norm = MinMaxScaler().fit(X)
# transform data
X = norm.transform(X)

print(X)