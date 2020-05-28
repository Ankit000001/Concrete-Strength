# This workflow will build and push a node.js application to an Azure Web App when a release is created.
#
# This workflow assumes you have already created the target Azure App Service web app.
# For instructions see https://docs.microsoft.com/azure/app-service/app-service-plan-manage#create-an-app-service-plan
#
# To configure this workflow:
#
# 1. Set up a secret in your repository named AZURE_WEBAPP_PUBLISH_PROFILE with the value of your Azure publish profile.
#    For instructions on obtaining the publish profile see: https://docs.microsoft.com/azure/app-service/deploy-github-actions#configure-the-github-secret
#
# 2. Change the values for the AZURE_WEBAPP_NAME, AZURE_WEBAPP_PACKAGE_PATH and NODE_VERSION environment variables  (below).
#
# For more information on GitHub Actions for Azure, refer to https://github.com/Azure/Actions
# For more samples to get started with GitHub Action workflows to deploy to Azure, refer to https://github.com/Azure/actions-workflow-samples
on:
  release:
    types: [created]

env:
  AZURE_WEBAPP_NAME: your-app-name    # set this to your application's name
  AZURE_WEBAPP_PACKAGE_PATH: '.'      # set this to the path to your web app project, defaults to the repository root
  NODE_VERSION: '10.x'                # set this to the node version to use

# -*- coding: utf-8 -*
import numpy as np           #Abbreviation for numpy array as i.e. alias,to play with arrays of columns in dataset
import pandas as pd          #Abbreviation for pandas library which is used to load dataset
#Loading libraries
#from pandas import read_csv     #to read csv format datafiles but no need as we've already imported whole pandas
from pandas.plotting import scatter_matrix
from matplotlib import pyplot      #Plotting various simpler plots
import seaborn as sns              #Nicely formatted plot,its from matplotlib
sns.set()                          #Now sns is ready to use using pyplot
from sklearn.model_selection import train_test_split   #to separate testing & training dataset
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
#Loading Dataset
names=['cmt','slg','fly','wat','plas','ca','fa','age','cst']     #Name of columns
#cement,blast furnace slag,fly ash,water,superplasticizer,coarse aggregate,fine aggregate,age,compressive strength
dataset = pd.read_csv("mycs.csv",names=names)    #Reading/importing of dataset,csv file should be on same directry path as specified in preferences,here on desktop
print(dataset)          #Printing in consol window of whole imported dataset 
#Storing various column in numpy arrays in case needed later
np_cmt=np.array(dataset.cmt)
np_slg=np.array(dataset.slg)
np_fly=np.array(dataset.fly)
np_wat=np.array(dataset.wat)
np_plas=np.array(dataset.plas)
np_ca=np.array(dataset.ca)
np_fa=np.array(dataset.fa)
np_age=np.array(dataset.age)
np_cst=np.array(dataset.cst)
print(np.mean(np_cst))          #Mean of response or output variable cst
#Summarizing the Dataset
#Dimension of Dataset
print(dataset.shape)
#All information about dataset
print(dataset.info())
#Eye on Data
print(dataset.head(20))
#Statistical Summary
print(dataset.describe())
#Class Distribution
print(dataset.groupby('cst').size())

#Data Visualization
## Univariate plots
#Box & Whisker plots
dataset.plot(kind='box',subplots=True,layout=(1,9),sharex=False,sharey=False,figsize=[25,15])
pyplot.show()
#Histogram                          #To show we've taken too much variety of data & it is uniform data collection as well
dataset.hist(layout=(2,5),figsize=[20,15])          #layout is used to avoid conjetion,i.e. 2 rows & 5 col,combined histogram
pyplot.show()
_=pyplot.hist(np_cmt,bins=20,color='Gray')        #Default bins(no of vertical bars) is 10 in matplotlib
_=pyplot.xlabel('Cement in kg')
pyplot.show()
_=pyplot.hist(np_slg,bins=20,color='Brown')
_=pyplot.xlabel('Blast Furnace slag in kg')
pyplot.show()
_=pyplot.hist(np_fly,bins=20,color='White')
_=pyplot.xlabel('Fly Ash in kg')
pyplot.show()
_=pyplot.hist(np_wat,bins=20,color='Blue')
_=pyplot.xlabel('Water in kg')
pyplot.show()
_=pyplot.hist(np_plas,bins=20,color='Black')
_=pyplot.xlabel('Superplasticizer in kg')
pyplot.show()
_=pyplot.hist(np_ca,bins=20,color='Skyblue')
_=pyplot.xlabel('Coarse Aggregate in kg')
pyplot.show()
_=pyplot.hist(np_fa,bins=20,color='Orange')
_=pyplot.xlabel('Fine Aggregate in kg')
pyplot.show()
_=pyplot.hist(np_age,bins=20,color='Yellow')
_=pyplot.xlabel('Age of testing in days')
pyplot.show()
_=pyplot.hist(np_cst,bins=20,color='Red')
_=pyplot.xlabel('Compressive Strength in MPa')
pyplot.show()
#ECDF                  #To show how small magnitude or how big magnitude features are taken ,its like grain size distribution analysis curve in soil,from this we can also read every percentile (including median) value
def ecdf(arr):         #It is manually defined function to obtain x & y inputs to ecdf plot 
    x=np.sort(arr)
    n=len(arr)
    y = np.arange(1, len(arr)+1) / n
    return x, y

x_cmt,y_cmt=ecdf(np_cmt)
x_slg,y_slg=ecdf(np_slg)
x_fly,y_fly=ecdf(np_fly)
x_wat,y_wat=ecdf(np_wat)
x_plas,y_plas=ecdf(np_plas)
x_ca,y_ca=ecdf(np_ca)
x_fa,y_fa=ecdf(np_fa)
x_age,y_age=ecdf(np_age)
x_cst,y_cst=ecdf(np_cst)
#ECDF of input features
_ = pyplot.plot(x_cmt, y_cmt,color='gray', marker='*')
_ = pyplot.plot(x_slg, y_slg,color='brown', marker='.')
_ = pyplot.plot(x_fly, y_fly,color='white', marker='.')
_ = pyplot.plot(x_wat, y_wat,color='blue', marker='*')
_ = pyplot.plot(x_plas, y_plas,color='black', marker='.')
_ = pyplot.plot(x_ca, y_ca, color='skyblue',marker='*')
_ = pyplot.plot(x_fa, y_fa, color='orange',marker='*')
_ = pyplot.plot(x_age, y_age,color='yellow', marker='.')
_ = pyplot.legend(('cement', 'Blast Furnace slag', 'Fly Ash','Water','SuperPlasticizer','Coarse Aggregate','Fine Aggregate','age'), loc='lower right')
_=pyplot.xlabel('Feature or Predictor Variables')
_=pyplot.ylabel('ECDF')
pyplot.show()
#ECDF of output Variable
_ = pyplot.plot(x_cst, y_cst,color='red', marker='D')
_=pyplot.xlabel('Output or Response Variable=Compressive Strength')
_=pyplot.ylabel('ECDF or Cummulative Frequency')
pyplot.show()
## Multivariate plots
#Bee Sworm Plots
_=sns.swarmplot(x='age',y='cst',data=dataset,color='yellow')   #To show even in less age cst can be a bit higher depending upon proportion of various ingredients,but it is still relatively lesser than more days cst
pyplot.show()
_=sns.swarmplot(x='plas',y='wat',data=dataset,color='skyblue')   #To show more the plasticizer less the water is consumed,it proves validity of chosen dataset 
pyplot.show()
#Scatter plot
scatter_matrix(dataset,figsize=[20,20],diagonal="kde",alpha=0.4,grid=True,marker='*')    #figsize is used to increase size of grids & kde gives better view than default histogram for diagonal elements 
pyplot.show()
#Relplot(3 variate)
sns.relplot(x='cmt',y='cst',hue='wat',alpha=0.5,data=dataset)     #Effect of w/c ratio on compressive strength
sns.relplot(x='ca',y='cst',hue='cmt',alpha=0.5,data=dataset)      #when ca is very less more cement is needed & cst is comparatively less,when ca is very much then also due to bad gradation cement is more needed but high cst


#Algorithms

##1.Separating the validation dataset          #Let last 20% of read dataset,we take as validation dataset,validation dataset is the one on which we will test accuracy later on
array= dataset.values                 #dataframe(list) to array conversion
X=array[:,0:-1]                       #Array splicing,all rows & all columns except last one(cst,because it is x & cst is result)  #indice -1 is equal to last col(here -1=8)...this is called negative indices
Y=array[:,-1]                         #All rows & only last column,i.e. cst=output column
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.2)     #use of train_test_split as per its syntax

#print(X_train,Y_train)              #To cross validate manually that split b/w training & validation dataset is done in disired way & same as it was in csv file
#----------------------------Review & retry from here------------------------
##2.K fold harness
   #  kfold=StratifiedKFold(n_splits=10,random_state=1,sh￼uffle=True) #In every K=10 no. of data K-1 are training & last one is testing

##3.Building the models               #from sklearn using predefined models 
#models=[]
#Logistic regression
#models.append(('LR',LogisticRegression(solver='liblinear',multi_class='ovr')))
#Linear Discriminant Analysis
#models.append(('LDA',LinearDiscriminantAnalysis()))
#K-Nearest Neighbors
#models.append(('KNN',KNeighborsClassifier()))
#Classification & regression trees
#models.append(('CART',DecisionTreeClassifier()))
#Gaussian Naive Bayes
#models.append(('NB',GaussianNB()))
#Support Vector Machines
#models.append(('SVM',SVC(gamma='auto')))

#Evaluating each model
#results=[]
#names=[]

#Creating results for Selecting Best Model
#for name,model in models:                   #Iteration for executing each model one by one
   # kfold=StratifiedKFold(n_splits=10,random_state=1,shuffle=True)
   # cv_results=cross_val_score(model,X_train,Y_train,cv=kfold,scoring='accuracy')
#results.append(cv_results)
#names.append(name)
#print('%s: %f \t %f'%(name,cv_results.mean(),cv_results.std()))
#Comparing Algorithms
#pyplot.boxplot(results,labels=names)
#pyplot.show()

#LinearRegression Model(Unregularized)
lnreg=LinearRegression()       #Initiallizing LinearRegression to a variable lnreg
lnreg.fit(X_train,Y_train)                            #Fit the model on training dataset
lnreg_pred=lnreg.predict(X_test)                   #Prediction on test data & storing these predictions in a variable lnreg_pred
lnregscore=lnreg.score(X_test,Y_test)                    #This is by default R^2 score####WITHOUT REGULARIZATION
print("R2 of LinearRegression is:",lnregscore)                                     #It is R^2 of not regularized linear regression

#Ridge Regression1(Regularized regression model,it uses alpha value(this control complexity of model,i.e. coeffi. are too large or too low,i.e. overfit or underfit) obtained by hyperparameter tuning,here firstly using a random alpha and seeing score)
ridge1=Ridge(alpha=0.5,normalize=True)    #initiallizing Ridge to a variable ridge1,it is without hyperparameter tuning & alpha is taken randomly as 0.5
ridge1.fit(X_train,Y_train)
ridge1_pred=ridge1.predict(X_test)
ridge1score=ridge1.score(X_test,Y_test)
print("R2 of Ridge without hyperparameter tuning is:",ridge1score)

#Lasso Regression1(without hyperparameter tuning)
lasso1=Lasso(alpha=0.5,normalize=True)
lasso1.fit(X_train,Y_train)
lasso1_pred=lasso1.predict(X_test)
lasso1score=lasso1.score(X_test,Y_test)
print("R2 of Lasso without hyperparameter tuning is:",lasso1score)

#Hyperparameter Tuning(selecting best value for hyperparameter,for this we also do cross validation together,so that best hyperperameter is not limited to some data only)
#1.Hyperparameter tuning for RidgeRegression,(here hyperparameter is only 1,alpha)

jobs:
  build-and-deploy:
    name: Build and Deploy
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Use Node.js ${{ env.NODE_VERSION }}
      uses: actions/setup-node@v1
      with:
        node-version: ${{ env.NODE_VERSION }}
    - name: npm install, build, and test
      run: |
        # Build and test the project, then
        # deploy to Azure Web App.
        npm install
        npm run build --if-present
        npm run test --if-present
    - name: 'Deploy to Azure WebApp'
      uses: azure/webapps-deploy@v2
      with:
        app-name: ${{ env.AZURE_WEBAPP_NAME }}
        publish-profile: ${{ secrets.AZURE_WEBAPP_PUBLISH_PROFILE }}
        package: ${{ env.AZURE_WEBAPP_PACKAGE_PATH }}
