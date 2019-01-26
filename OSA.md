

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
osa=pd.read_excel('path/osa.xlsx')
```


```python
osa.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Number</th>
      <th>GENDER</th>
      <th>RACE</th>
      <th>stop</th>
      <th>stopA</th>
      <th>stopbang</th>
      <th>stopbangA</th>
      <th>OSA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>5</td>
      <td>7</td>
      <td>8</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>6</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>6</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### First use a decision tree

Notice here I did not do train_test split, that is
because I only want to see if using decision tree method
can perform better than only use the stopbangA

more data should be collected to test the predicted result


```python
from sklearn.tree import DecisionTreeClassifier
```


```python
X=osa.drop(['OSA','Number'],axis=1)
Y=osa['OSA']
```


```python
dtree=DecisionTreeClassifier()
```


```python
dtree.fit(X,Y)
```




    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best')




```python
pred=dtree.predict(X)
```


```python
from sklearn.metrics import classification_report, confusion_matrix
```


```python
print(classification_report(Y,pred))
```

                 precision    recall  f1-score   support
    
              0       0.79      0.76      0.77        29
              1       0.93      0.94      0.94       105
    
    avg / total       0.90      0.90      0.90       134
    
    


```python
print(confusion_matrix(Y,pred))
```

    [[22  7]
     [ 6 99]]
    

Now try a random forest


```python
from sklearn.ensemble import RandomForestClassifier
```

    D:\Python\Python\anaconda\lib\site-packages\sklearn\ensemble\weight_boosting.py:29: DeprecationWarning: numpy.core.umath_tests is an internal NumPy module and should not be imported. It will be removed in a future NumPy release.
      from numpy.core.umath_tests import inner1d
    


```python
rft=RandomForestClassifier(n_estimators=100)
```


```python
rft.fit(X,Y)
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)




```python
predict=rft.predict(X)
```


```python
print(confusion_matrix(Y,predict))
```

    [[ 18  11]
     [  2 103]]
    


```python
print(X.columns,rft.feature_importances_)
```

    Index(['GENDER', 'RACE', 'stop', 'stopA', 'stopbang', 'stopbangA'], dtype='object') [0.11571472 0.1237057  0.11767914 0.15902426 0.24939356 0.23448262]
    


```python
print(classification_report(Y,predict))
```

                 precision    recall  f1-score   support
    
              0       0.90      0.62      0.73        29
              1       0.90      0.98      0.94       105
    
    avg / total       0.90      0.90      0.90       134
    
    

#### The resluts above show that decision tree or random forest all can better fit the data than only stopbangA, but a random forest seems not improve the outcome significantly comparing to the decision tree
