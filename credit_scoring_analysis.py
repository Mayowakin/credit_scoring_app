import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt

data = pd.read_csv('bank.csv',sep=';')

data.head()

data.columns

data.describe()

data.dtypes

data.isnull().sum()

data.info()

# Box plot for job and balance
import plotly.express as px
fig = px.box(data,x='job',y = 'balance',color='job', title='Balance/Job')
fig.update_layout(title ={ 
    'x' : 0.5,
    'y' : 0.9,
    'xanchor': 'center',
    'yanchor': 'top'
    })
fig.show()

# Pie chart for identify the ratio of every marital state 
marital_chart = data['marital'].value_counts()
plt.pie(marital_chart,labels=marital_chart.index,autopct='%.1f%%')

marital_chart = data['housing'].value_counts()
plt.pie(marital_chart,labels=marital_chart.index,autopct='%.1f%%')

month_pie = data['month'].value_counts()
plt.pie(month_pie.values,labels=month_pie.index)

educatin_bar = data['education'].value_counts()
plt.bar(x = educatin_bar.index,height=educatin_bar.values)

contact_chart = data['contact'].value_counts()
plt.bar(x = contact_chart.index, height = contact_chart.values,)
plt.show()

contact_chart = data['poutcome'].value_counts()
plt.pie(contact_chart.values, labels =  contact_chart.index,)
plt.show()

sns.pairplot(data,height=3)
plt.suptitle('pair plot of features')
plt.show()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == object:
        data[col] = le.fit_transform(data[col])

corr = data.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr,annot=True,cmap = 'RdBu',fmt = ".2f")

# dividing data to the dependent and independent vars
x = data.drop('y',axis=1)
y= data['y']

# training classificaton models

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

# feature scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

from sklearn.metrics import accuracy_score,confusion_matrix

def classifier_acc(classifier):
    # Fit the classifier
    classifier.fit(x_train, y_train)
    
    # Predict on the test set
    y_pred = classifier.predict(x_test)
    
    # Calculate accuracy and confusion matrix
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    return acc, cm

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

classifiers = [SVC(), 
               LogisticRegression(), 
               RandomForestClassifier(), 
               GradientBoostingClassifier(),
               DecisionTreeClassifier(), 
               KNeighborsClassifier(),
               GaussianNB()
              ]
results = []
for classifier in classifiers:
    acc,cm = classifier_acc(classifier)
    results.append({
        'classifier_algorithm':type(classifier).__name__,
        'accurecy' : acc  ,
        'confusion matrix':cm}
                  )

print(f'{results} \n')

max_accurecy = 0
max_classifier_accurecy = 'logistic'
for ele in results:
    if ele['accurecy']>max_accurecy:
        max_accurecy = ele['accurecy']
        max_classifier_accurecy = ele['classifier_algorithm']

print(f'the max accurey : {max_accurecy*100 : .2f} % when using : {max_classifier_accurecy} classifier algorithm ')