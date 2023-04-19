```python
# import libraries

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import random


from sklearn import datasets
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score #repeatly split the train and test data
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LogisticRegression
```


```python
#load dataset Breast Cancer

Loan_df = pd.read_excel('D:/Mahdieh_CourseUniversity/University_courses/ALY6020/Module_3/Mid_Week/Bank_Personal_Loan_Modelling.xlsx')
Loan_df
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
      <th>ID</th>
      <th>Age</th>
      <th>Experience</th>
      <th>Income</th>
      <th>ZIP Code</th>
      <th>Family</th>
      <th>CCAvg</th>
      <th>Education</th>
      <th>Mortgage</th>
      <th>Personal Loan</th>
      <th>Securities Account</th>
      <th>CD Account</th>
      <th>Online</th>
      <th>CreditCard</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>25</td>
      <td>1</td>
      <td>49</td>
      <td>91107</td>
      <td>4</td>
      <td>1.6</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>45</td>
      <td>19</td>
      <td>34</td>
      <td>90089</td>
      <td>3</td>
      <td>1.5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>39</td>
      <td>15</td>
      <td>11</td>
      <td>94720</td>
      <td>1</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>35</td>
      <td>9</td>
      <td>100</td>
      <td>94112</td>
      <td>1</td>
      <td>2.7</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>35</td>
      <td>8</td>
      <td>45</td>
      <td>91330</td>
      <td>4</td>
      <td>1.0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4995</th>
      <td>4996</td>
      <td>29</td>
      <td>3</td>
      <td>40</td>
      <td>92697</td>
      <td>1</td>
      <td>1.9</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4996</th>
      <td>4997</td>
      <td>30</td>
      <td>4</td>
      <td>15</td>
      <td>92037</td>
      <td>4</td>
      <td>0.4</td>
      <td>1</td>
      <td>85</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4997</th>
      <td>4998</td>
      <td>63</td>
      <td>39</td>
      <td>24</td>
      <td>93023</td>
      <td>2</td>
      <td>0.3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4998</th>
      <td>4999</td>
      <td>65</td>
      <td>40</td>
      <td>49</td>
      <td>90034</td>
      <td>3</td>
      <td>0.5</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4999</th>
      <td>5000</td>
      <td>28</td>
      <td>4</td>
      <td>83</td>
      <td>92612</td>
      <td>3</td>
      <td>0.8</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5000 rows × 14 columns</p>
</div>




```python
# Descriptive analysis

Loan_df.head()
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
      <th>ID</th>
      <th>Age</th>
      <th>Experience</th>
      <th>Income</th>
      <th>ZIP Code</th>
      <th>Family</th>
      <th>CCAvg</th>
      <th>Education</th>
      <th>Mortgage</th>
      <th>Personal Loan</th>
      <th>Securities Account</th>
      <th>CD Account</th>
      <th>Online</th>
      <th>CreditCard</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>25</td>
      <td>1</td>
      <td>49</td>
      <td>91107</td>
      <td>4</td>
      <td>1.6</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>45</td>
      <td>19</td>
      <td>34</td>
      <td>90089</td>
      <td>3</td>
      <td>1.5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>39</td>
      <td>15</td>
      <td>11</td>
      <td>94720</td>
      <td>1</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>35</td>
      <td>9</td>
      <td>100</td>
      <td>94112</td>
      <td>1</td>
      <td>2.7</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>35</td>
      <td>8</td>
      <td>45</td>
      <td>91330</td>
      <td>4</td>
      <td>1.0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
Loan_df.shape
```




    (5000, 14)




```python
Loan_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5000 entries, 0 to 4999
    Data columns (total 14 columns):
     #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
     0   ID                  5000 non-null   int64  
     1   Age                 5000 non-null   int64  
     2   Experience          5000 non-null   int64  
     3   Income              5000 non-null   int64  
     4   ZIP Code            5000 non-null   int64  
     5   Family              5000 non-null   int64  
     6   CCAvg               5000 non-null   float64
     7   Education           5000 non-null   int64  
     8   Mortgage            5000 non-null   int64  
     9   Personal Loan       5000 non-null   int64  
     10  Securities Account  5000 non-null   int64  
     11  CD Account          5000 non-null   int64  
     12  Online              5000 non-null   int64  
     13  CreditCard          5000 non-null   int64  
    dtypes: float64(1), int64(13)
    memory usage: 547.0 KB
    


```python
Loan_df.describe()
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
      <th>ID</th>
      <th>Age</th>
      <th>Experience</th>
      <th>Income</th>
      <th>ZIP Code</th>
      <th>Family</th>
      <th>CCAvg</th>
      <th>Education</th>
      <th>Mortgage</th>
      <th>Personal Loan</th>
      <th>Securities Account</th>
      <th>CD Account</th>
      <th>Online</th>
      <th>CreditCard</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.00000</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2500.500000</td>
      <td>45.338400</td>
      <td>20.104600</td>
      <td>73.774200</td>
      <td>93152.503000</td>
      <td>2.396400</td>
      <td>1.937913</td>
      <td>1.881000</td>
      <td>56.498800</td>
      <td>0.096000</td>
      <td>0.104400</td>
      <td>0.06040</td>
      <td>0.596800</td>
      <td>0.294000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1443.520003</td>
      <td>11.463166</td>
      <td>11.467954</td>
      <td>46.033729</td>
      <td>2121.852197</td>
      <td>1.147663</td>
      <td>1.747666</td>
      <td>0.839869</td>
      <td>101.713802</td>
      <td>0.294621</td>
      <td>0.305809</td>
      <td>0.23825</td>
      <td>0.490589</td>
      <td>0.455637</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>23.000000</td>
      <td>-3.000000</td>
      <td>8.000000</td>
      <td>9307.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1250.750000</td>
      <td>35.000000</td>
      <td>10.000000</td>
      <td>39.000000</td>
      <td>91911.000000</td>
      <td>1.000000</td>
      <td>0.700000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2500.500000</td>
      <td>45.000000</td>
      <td>20.000000</td>
      <td>64.000000</td>
      <td>93437.000000</td>
      <td>2.000000</td>
      <td>1.500000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3750.250000</td>
      <td>55.000000</td>
      <td>30.000000</td>
      <td>98.000000</td>
      <td>94608.000000</td>
      <td>3.000000</td>
      <td>2.500000</td>
      <td>3.000000</td>
      <td>101.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5000.000000</td>
      <td>67.000000</td>
      <td>43.000000</td>
      <td>224.000000</td>
      <td>96651.000000</td>
      <td>4.000000</td>
      <td>10.000000</td>
      <td>3.000000</td>
      <td>635.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.00000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#checking for missing values

Loan_df.isnull().sum()
```




    ID                    0
    Age                   0
    Experience            0
    Income                0
    ZIP Code              0
    Family                0
    CCAvg                 0
    Education             0
    Mortgage              0
    Personal Loan         0
    Securities Account    0
    CD Account            0
    Online                0
    CreditCard            0
    dtype: int64




```python
Loan_df.isnull().any()
```




    ID                    False
    Age                   False
    Experience            False
    Income                False
    ZIP Code              False
    Family                False
    CCAvg                 False
    Education             False
    Mortgage              False
    Personal Loan         False
    Securities Account    False
    CD Account            False
    Online                False
    CreditCard            False
    dtype: bool




```python
plt.figure(figsize=(6,4))
sns.heatmap(Loan_df.isnull())
```




    <AxesSubplot: >




    
![png](output_8_1.png)
    


ID coulumn in our database have a unique number for every client. it is not needed and can be dropped.


```python
Loan_df.drop(['ID'], axis = 1, inplace=True)

```

we want to perform Logistic Regression, hence we check assumptions for logistic regression :
Basic assumptions that must be met for logistic regression include independence of errors, linearity in the logit for continuous variables, absence of multicollinearity, and lack of strongly influential outliers.


```python
plt.figure(figsize=(20,8))
Loan_df.boxplot()
```




    <AxesSubplot: >




    
![png](output_12_1.png)
    



```python
# we use correlatiion to check multicollinearity
plt.figure(figsize=(12,10))
sns.heatmap(Loan_df.corr(), annot= True)
```




    <AxesSubplot: >




    
![png](output_13_1.png)
    



```python
# Age and experience are highly correlated and one of them can be dropped, I dropped experience.
# Since experience had negative values dropping experience would be better option

Loan_df.drop(['Experience'], axis = 1, inplace=True)
```


```python
Loan_df
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
      <th>Age</th>
      <th>Income</th>
      <th>ZIP Code</th>
      <th>Family</th>
      <th>CCAvg</th>
      <th>Education</th>
      <th>Mortgage</th>
      <th>Personal Loan</th>
      <th>Securities Account</th>
      <th>CD Account</th>
      <th>Online</th>
      <th>CreditCard</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>25</td>
      <td>49</td>
      <td>91107</td>
      <td>4</td>
      <td>1.6</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>45</td>
      <td>34</td>
      <td>90089</td>
      <td>3</td>
      <td>1.5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>39</td>
      <td>11</td>
      <td>94720</td>
      <td>1</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35</td>
      <td>100</td>
      <td>94112</td>
      <td>1</td>
      <td>2.7</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35</td>
      <td>45</td>
      <td>91330</td>
      <td>4</td>
      <td>1.0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4995</th>
      <td>29</td>
      <td>40</td>
      <td>92697</td>
      <td>1</td>
      <td>1.9</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4996</th>
      <td>30</td>
      <td>15</td>
      <td>92037</td>
      <td>4</td>
      <td>0.4</td>
      <td>1</td>
      <td>85</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4997</th>
      <td>63</td>
      <td>24</td>
      <td>93023</td>
      <td>2</td>
      <td>0.3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4998</th>
      <td>65</td>
      <td>49</td>
      <td>90034</td>
      <td>3</td>
      <td>0.5</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4999</th>
      <td>28</td>
      <td>83</td>
      <td>92612</td>
      <td>3</td>
      <td>0.8</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5000 rows × 12 columns</p>
</div>



Perform Logistic Regression


```python
# indicate predictor and predicted variables

y = Loan_df['Personal Loan']

x = Loan_df.drop(['Personal Loan'], axis=1)
```


```python
# split our data into training and testing data

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
```


```python
#Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
```


```python
#Perform model

logistic_model=LogisticRegression(solver='liblinear', random_state= 1 )
logistic_model.fit(x_train, y_train)
y_pred = logistic_model.predict(x_test)
cm=confusion_matrix(y_test,y_pred)

print(logistic_model.score(x_train,y_train)*100)
print(accuracy_score(y_test, y_pred)*100)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

    95.19999999999999
    95.39999999999999
    [[889   9]
     [ 37  65]]
                  precision    recall  f1-score   support
    
               0       0.96      0.99      0.97       898
               1       0.88      0.64      0.74       102
    
        accuracy                           0.95      1000
       macro avg       0.92      0.81      0.86      1000
    weighted avg       0.95      0.95      0.95      1000
    
    


```python
model = sm.Logit(y_train,x_train).fit()
print(model.summary())
```

    Optimization terminated successfully.
             Current function value: 0.620463
             Iterations 6
                               Logit Regression Results                           
    ==============================================================================
    Dep. Variable:          Personal Loan   No. Observations:                 4000
    Model:                          Logit   Df Residuals:                     3989
    Method:                           MLE   Df Model:                           10
    Date:                Sun, 29 Jan 2023   Pseudo R-squ.:                 -0.9834
    Time:                        20:22:01   Log-Likelihood:                -2481.9
    converged:                       True   LL-Null:                       -1251.3
    Covariance Type:            nonrobust   LLR p-value:                     1.000
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    x1             0.0186      0.034      0.545      0.586      -0.048       0.085
    x2             0.6173      0.049     12.619      0.000       0.521       0.713
    x3             0.0052      0.034      0.154      0.878      -0.061       0.071
    x4             0.1838      0.035      5.283      0.000       0.116       0.252
    x5             0.1417      0.047      3.022      0.003       0.050       0.234
    x6             0.2998      0.035      8.462      0.000       0.230       0.369
    x7             0.0296      0.037      0.805      0.421      -0.042       0.102
    x8            -0.1062      0.038     -2.816      0.005      -0.180      -0.032
    x9             0.4759      0.050      9.522      0.000       0.378       0.574
    x10           -0.0675      0.034     -1.968      0.049      -0.135      -0.000
    x11           -0.0922      0.036     -2.582      0.010      -0.162      -0.022
    ==============================================================================
    


```python
plt.figure(figsize=(10,6))
plt.title('Confusion Matrix')
sns.heatmap(cm, annot=True,fmt='d', cmap='Blues')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
```




    Text(0.5, 36.72222222222221, 'Predicted Values')




    
![png](output_22_1.png)
    



```python
# Optimize the model by droping unsignificentt variables 

Loan_df.drop(['Age' , 'Mortgage' , 'ZIP Code','Online'] , axis= 1 , inplace=True) 
```


```python
# indicate predictor and predicted variables

y = Loan_df['Personal Loan']

x = Loan_df.drop(['Personal Loan'], axis=1)
```


```python
x
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
      <th>Income</th>
      <th>Family</th>
      <th>CCAvg</th>
      <th>Education</th>
      <th>Securities Account</th>
      <th>CD Account</th>
      <th>CreditCard</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>49</td>
      <td>4</td>
      <td>1.6</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>34</td>
      <td>3</td>
      <td>1.5</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11</td>
      <td>1</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>100</td>
      <td>1</td>
      <td>2.7</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>45</td>
      <td>4</td>
      <td>1.0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4995</th>
      <td>40</td>
      <td>1</td>
      <td>1.9</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4996</th>
      <td>15</td>
      <td>4</td>
      <td>0.4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4997</th>
      <td>24</td>
      <td>2</td>
      <td>0.3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4998</th>
      <td>49</td>
      <td>3</td>
      <td>0.5</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4999</th>
      <td>83</td>
      <td>3</td>
      <td>0.8</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5000 rows × 7 columns</p>
</div>




```python
# split our data into training and testing data

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
```


```python
#Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
```


```python
#Perform model

logistic_model=LogisticRegression(solver='liblinear', random_state= 1 )
logistic_model.fit(x_train, y_train)
y_pred = logistic_model.predict(x_test)
cm=confusion_matrix(y_test,y_pred)

print(logistic_model.score(x_train,y_train)*100)
print(accuracy_score(y_test, y_pred)*100)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

    95.14285714285714
    94.93333333333334
    [[1340   13]
     [  63   84]]
                  precision    recall  f1-score   support
    
               0       0.96      0.99      0.97      1353
               1       0.87      0.57      0.69       147
    
        accuracy                           0.95      1500
       macro avg       0.91      0.78      0.83      1500
    weighted avg       0.95      0.95      0.94      1500
    
    


```python
model = sm.Logit(y_train,x_train).fit()

print(model.summary())
```


```python
plt.figure(figsize=(10,6))
plt.title('Confusion Matrix')
sns.heatmap(cm, annot=True,fmt='d', cmap='Blues')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
```




    Text(0.5, 36.72222222222221, 'Predicted Values')




    
![png](output_30_1.png)
    

