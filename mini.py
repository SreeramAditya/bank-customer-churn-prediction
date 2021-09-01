import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
sns.set()


df = pd.read_csv('Churn_Modelling.csv')
df.head()
df.shape
df.dtypes
print(df['Geography'].unique())
print(df['Gender'].unique())
print(df['NumOfProducts'].unique())
print(df['HasCrCard'].unique())
print(df['IsActiveMember'].unique())
df.isnull().sum()
df.describe()   
final_dataset = df[['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited']]
final_dataset.head()
final_dataset = pd.get_dummies(final_dataset, drop_first=True)
final_dataset.head()
corrmat = final_dataset.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
sns.heatmap(final_dataset[top_corr_features].corr(), annot=True, cmap='RdYlGn')
final_dataset.head()
X = final_dataset.iloc[:, [0,1,2,3,4,5,6,7,9,10,11]]
y = final_dataset.iloc[:, 8].values
X.head()
y
#logistic regression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state = 42)
x1=sm.add_constant(X_train)
reg_log=sm.Logit(y_train,X_train)
results_log=reg_log.fit()
results_log.summary()
results_log.predict()
np.array(final_dataset['Exited'])
results_log.pred_table()
cm_df=pd.DataFrame(results_log.pred_table())
cm_df.columns=['Predict 0','Predict 1']
cm_df=cm_df.rename(index={0:'Actual 0',1:'Actual 1'})
cm_df
cm=np.array(cm_df)
accuracy_train=(cm[0,0]+cm[1,1])/cm.sum()
accuracy_train
def confusion_matrix(final_dataset,actual_values,model):
    pred_values=model.predict(final_dataset)
    bins=np.array([0,0.5,1])
    cm=np.histogram2d(actual_values,pred_values,bins=bins)[0]
    accuracy=(cm[0,0]+cm[1,1])/cm.sum()
    return cm,accuracy
cm=confusion_matrix(X_test,y_test,results_log)
cm
#random forest classifier
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train)
from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(X,y)
print(model.feature_importances_)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)
print(accuracy_score(y_test,y_pred))
file= open('Customer.pkl','wb')
pickle.dump(rf,file)