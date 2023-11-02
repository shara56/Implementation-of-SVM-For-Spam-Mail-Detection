# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.
2. Import the dataset to operate on.
3. Split the dataset.
4. Predict the required output.
5. End the program.
 

## Program:
```

Program to implement the SVM For Spam Mail Detection..
Developed by: SHARANGINI T K
RegisterNumber:  212222230143


import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result


import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

```

## Output:

### Result Output
![1](https://github.com/hariprasath5106/Implementation-of-SVM-For-Spam-Mail-Detection/assets/111515488/2ee73eb9-9122-4bbe-a23c-b3dc6b77e870)

### data.head( )
![2](https://github.com/hariprasath5106/Implementation-of-SVM-For-Spam-Mail-Detection/assets/111515488/f881ca6c-0838-41c6-ab4d-53e38016911e)

### data.info( )
![3](https://github.com/hariprasath5106/Implementation-of-SVM-For-Spam-Mail-Detection/assets/111515488/bc457dd4-2cf4-4bc9-a57f-f31370925bf5)

## data.isnull().sum()
![4](https://github.com/hariprasath5106/Implementation-of-SVM-For-Spam-Mail-Detection/assets/111515488/ecc04c7e-7e51-4539-b588-796875b3da9b)

### Y_prediction
![5](https://github.com/hariprasath5106/Implementation-of-SVM-For-Spam-Mail-Detection/assets/111515488/45b6bf1f-f929-417c-9239-bcb82827824c)

### Accuracy Value
![6](https://github.com/hariprasath5106/Implementation-of-SVM-For-Spam-Mail-Detection/assets/111515488/c7d01baa-d8ff-4bde-9904-4bbcfcad44b6)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
