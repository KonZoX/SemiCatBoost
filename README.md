# Semisupervised CatBoost
A wrapper for sklearn's SelfTrainingClassifier that supports CatBoost

## Data sources:
[kaggle](https://www.kaggle.com/competitions/spaceship-titanic/submissions)
## Target
- labels denoting positive, negative class of survival


- 
```sh
import pandas as pd
import numpy as np
from semi_classes import SelfTrainingClassifierExtended as SelfTraining, CatBoostClassifierExtended

df = pd.read_csv("/spaceship-titanic/train.csv")
df = df.sample(frac=1.0)

test = pd.read_csv("/home/constantz/Загрузки/spaceship-titanic/test.csv")
df = df.append(test)

df.CryoSleep = df.apply(lambda x : x.CryoSleep if x.CryoSleep==x.CryoSleep else
                       False if any([x.RoomService, x.FoodCourt, x.ShoppingMall, x.Spa, x.VRDeck])
                       else True, axis=1)
                       
target = df.Transported.values
y_mask = np.random.rand(df.shape[0] - test.shape[0]) < 0.15
for idx,x in enumerate(y_mask):
    if x == True:
        target[idx] = -1
        
target *= 1
target = np.array([-1 if np.isnan(x) else x for x in target ])

traindf = df.drop('Transported', axis=1).fillna(-999)
categorical_features_indices = np.where(traindf.dtypes != float)[0].tolist()

clf = CatBoostClassifierExtended(categorical_features_indices)
semiclf = SelfTraining(clf, verbose=True, threshold=0.75, criterion='threshold', k_best=100, max_iter=None)
semiclf.fit(traindf.values, target)

classes = semiclf.predict(traindf.values)
```

Expected f1-score 0.8




