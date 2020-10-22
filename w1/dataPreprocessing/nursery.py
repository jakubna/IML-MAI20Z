from scipy.io import arff
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder

# load dataset
f = 'C://Users//nadzi//Desktop//datasets//datasets//nursery.arff'
data, meta = arff.loadarff(f)
dataset = np.array(data.tolist(), dtype=object)

column_names = meta.names() # list containing column names
# create a initial pandas dataframe
df = pd.DataFrame(data=dataset, columns=column_names)

# detect and replace missing values
if df.isnull().any().sum() > 0:
    for x in meta[:-1]:
        top_frequent_value = df[x].describe()['top']
        df[x].fillna(top_frequent_value, inplace=True)

# split-out dataset
X = df.iloc[:,:-1]
Y = df.iloc[:, -1]

def label_encoding(x, classes):
    """Function written based on label encoding rule
        Parameters
        ----------
        x : value to interpret
        classes: array of classes
        Returns
        -------
        self : returns index of x value in classes array.
        """
    return classes.index(x)

# apply label_encoding function on selected columns
X.iloc[:,1] = X.iloc[:,1].apply(label_encoding, classes=[b'proper', b'less_proper', b'improper', b'critical', b'very_crit'])
X.iloc[:,3] = X.iloc[:,3].apply(label_encoding, classes=[b'1', b'2', b'3', b'more'])
X.iloc[:,4] = X.iloc[:,4].apply(label_encoding, classes=[b'convenient', b'less_conv', b'critical'])
X.iloc[:,5] = X.iloc[:,5].apply(label_encoding, classes=[b'convenient', b'inconv'])
X.iloc[:,6] = X.iloc[:,6].apply(label_encoding, classes=[b'nonprob', b'slightly_prob', b'problematic'])
X.iloc[:,7] = X.iloc[:,7].apply(label_encoding, classes=[b'not_recom', b'recommended', b'priority'])

# apply one-hot encoding using get_dummies function from pandas
X=pd.get_dummies(X, columns=['a1','a3'])

# fit MinMax scaler on data
norm = MinMaxScaler().fit(X)
# transform data
X = norm.transform(X)

print(X)