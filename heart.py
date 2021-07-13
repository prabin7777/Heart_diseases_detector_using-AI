# this is machine learning code for heart desiease calculator >>>>>
# github.com/prabin7777(Damodar pokharel)
# you can replace any csv datasheet by this code , really amazing 
#here i have used RandomforestClassifire from sklearn to train the model
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# # load daata
# from google.colab import files
# upload_files = files.upload()


df=pd.read_csv("heart.csv")
print(df.head(7))
a=pd.read_csv('1.csv')
#checking for nll value

df.isnull().values.any()
df.describe()
#checking for corelation

plt.figure(figsize=(7,7))
sns.heatmap(df.corr(), annot=True, fmt='.0%')
plt.show()

#split data into features and target data
X=df.iloc[:,:-1].values
Y=df.iloc[:,-1].values
D=a.iloc[:,:-1].values
#spliting 25% test data and 75% training data
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest= train_test_split(X,Y ,test_size=0.25, random_state=1)


#feature scaling
#scale the data with vvalue between 0 aand 1
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
xtrain=sc.fit_transform(xtrain)
xtest=sc.transform(xtest)
d=sc.transform(D)





#random frrsd classfire

from sklearn.ensemble import RandomForestClassifier
forest=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=1)
forest.fit(xtrain,ytrain)
model=forest
model.score(xtrain,ytrain)
#model training complete
#test model from 25% test data
from sklearn.metrics import confusion_matrix as com
cm=com(ytest,model.predict(xtest))
if(model.predict(d))==0:
    print("you donot have risk of heart disease")
else:
    print("you have risk of heart diseases")
TN=cm[0][0]# true negative
TP=cm[1][1]#true positive 
FN=cm[1][0]#false negative
FP=cm[0][1]#false positive
print(cm)
#for accuracy
accuracy=(TP+TN)*100/(TP+TN+FN+FP)
print("THE ACCURACY IS "+str(accuracy))# i have obtained 72% accuracy through this code 