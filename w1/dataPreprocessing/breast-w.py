from scipy.io import arff
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN

# load dataset
f = 'C://Users//nadzi//Desktop//datasets//datasets//breast-w.arff'
data, meta = arff.loadarff(f)
dataset = np.array(data.tolist(), dtype=object)
meta = meta.names() # list containing column names

# create a initial pandas dataframe
df = pd.DataFrame(data=dataset, columns=list(meta))

# detect missing values and replacing using median
if df.isnull().any().sum() > 0:
    for x in meta[:-1]:
        median = df[x].median()
        df[x].fillna(median, inplace=True)

# split-out dataset
X = df.iloc[:,:-1]
Y = df.iloc[:, -1]

# fit MinMax scaler on data
norm = MinMaxScaler().fit(X)
# transform data
X = norm.transform(X)

print(X)