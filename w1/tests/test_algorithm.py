from dataPreprocessing.breast import preprocess as preprocess_breast
from dataPreprocessing.cmc import preprocess as preprocess_cmc
from dataPreprocessing.nursery import preprocess as preprocess_nursery
from algorithms.KMeans import KMeans
from algorithms.KMedians import KMedians
from algorithms.FuzzyCMeans import FuzzyCMeans

X = preprocess_breast()
algorithm = FuzzyCMeans(k=2, max_it =10)
algorithm.fit(X)
a,b,c = algorithm.predict(X)
print(a)
print(b)
print(c)
