# Copyright 2021 Batyrkhan Altynbay
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Imports: 
import pandas as pd
import numpy as np
import scipy
import sklearn
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
from sklearn.ensemble import IsolationForest 
from sklearn.neighbors import LocalOutlierFactor 
from pyod.models.knn import KNN 
from pyod.models.ocsvm import OCSVM 
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize
from sklearn.preprocessing import StandardScaler
from cf_matrix import make_confusion_matrix 
import warnings 
warnings.filterwarnings('ignore')


#Loading Data 
data = pd.read_csv("creditcard.csv")



#Working File 
data.Class.value_counts(normalize=True)

corr = data.corr() 
fig = plt.figure(figsize=(30,20))
sns.heatmap(corr, vmax=.8, square=True,annot=True)


positive = data[data["Class"]== 1]
negative = data[data["Class"]== 0]


print("positive:{}".format(len(positive)))
print("negative:{}".format(len(negative)))

JE = pd.concat([positive,negative[:10000]])

JE = new_data.sample(frac=1,random_state=42)
JE.shape

JE['Amount'] = StandardScaler().fit_transform(JE['Amount'].values.reshape(-1,1))

X = JE.drop(['Time','Class'], axis=1) 
Y = JE['Class']  

 
X_train, X_test, y_train,y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=42 )

clf_knn = KNN(contamination=0.172, n_neighbors = 5,n_jobs=-1)

clf_knn.fit(X_train)