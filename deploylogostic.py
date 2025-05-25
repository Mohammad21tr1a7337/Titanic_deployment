
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


##creating a title for Deployment

st.set_page_config(page_title="🚢 Titanic Survival Predictor", layout="centered")
st.title("🚢 Titanic Survival Predictor")
st.markdown("### Would you have survived the Titanic disaster?")
st.markdown("Use this simulation based on a **logistic regression model** trained on Titanic data to predict your chances of survival.")


#creating the the input parameters

def user_input_feature():
    PassengerAge=st.sidebar.number_input("Age")
    PassengerGender=st.sidebar.selectbox('Gender Male(1),Female(0)',("0","1"))
    PassengerClass=st.sidebar.selectbox("Pclass 1st,2nd,3rd class",("1","2","3"))
    PassengerEmbarked=st.sidebar.selectbox("Embarked(C=Cherbourg,S=Southampton,Q=Queenstome)",("1","2","3"))
    PassengerSiblings=st.sidebar.selectbox("Siblings",("1","2","3","4","5","6"))
    PassengerParents=st.sidebar.selectbox("Parents",("1","2",'3','4','5','6','7'))
    data={"PassengerAge":PassengerAge,
          "PassengerGender":PassengerGender,
          "PassengerClass":PassengerClass,
          "PassengerEmbarked":PassengerEmbarked,
          "PassengerSiblings":PassengerSiblings,
          "PassengerParents":PassengerParents}
    features=pd.DataFrame(data,index=[0])
    return features

data=user_input_feature()
st.write("User Input Parameters")
st.write(data)

#===========================================================================================================================================================


dftrain=pd.read_csv("Titanic_train.csv")

#checking which imputation technique is best for the data

#data imputation 

dftrain['Cabin'].fillna('Unknown', inplace=True)    

dftrain.groupby('Ticket')['Cabin'].agg(lambda x: x.mode().iloc[0]) 

dftrain['Cabin'] = dftrain.groupby('Pclass')['Cabin'].transform(lambda x: x.fillna(x.mode()[0]))
                                       
dftrain['Cabin'].isnull().sum()

dftrain.isnull().sum()

dftrain = dftrain.dropna(subset=['Embarked'])  # Removes rows where 'Fare' is NaN

dftrain.isnull().sum()

#==================================================================================================================================================================

#we are performing the label Encoding on the categorical data

from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()       #storing the function in a instance

lab1=label.fit_transform(dftrain['Embarked'])   #transforming the data
lab1=pd.DataFrame(lab1)                         #Creating the data frame
lab1.columns=["Embarked"]                       #giving the name for the column


#==============================================================================================================================================================

#Colums to be Standardize or Normalize

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

scale=scaler.fit_transform(dftrain[['Age','Fare']])
scale=pd.DataFrame(scale)
scale.columns=['Age','Fare']

#==========================================================================================================================================================

#merging the all transformed columns in a single dataset

#train

df3=dftrain[['SibSp','Parch','Pclass','Survived']]

#df_train.dropna(inplace=True)  #checks for any null values and drop them
#df_train.isnull().sum()

#==============================================================================================================================================================

#creating a model

x=dftrain.columns[['Age','Sex','Pclass','SibSp','Parch']]  #independent variables using only pandas iloc func
y=dftrain['Survived']  #Dependent variable

from sklearn.linear_model import LogisticRegression  #import the model

model=LogisticRegression()     #creating a instance

log=model.fit(x,y)             #fitting the model

y_pred=model.predict(x)        #predicting on x data

y_pred_data=model.predict(data)
#finging the pridicted probability values

y_pred_prob=model.predict_proba(data)

#===================================================================================================================================================================

#printing the results using the streamlet the 


#accuracy_score(y,y_pred1)   # checking accuracy

st.subheader('Predicted Results Survived  Yes(1) No(0)')
st.write('Yes you will Survive In the Accident' if y_pred_prob[0][1] > 0.5 else "No you Won't Survive")

st.markdown("#### 🔍 Probability Breakdown")

st.subheader('Prediction Probability')
st.write(y_pred_prob)
