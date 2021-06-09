from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import tensorflow
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

st.title("Hear Failure Prediction Web App")
df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
st.write("""
## ***Prediction***:
""")
st.write("""
### Full the input in sidebar to set up  the algorithm that we can use
### **also**
### Full the input below to predict
""")


st.sidebar.title("Options for Prediction")
if st.sidebar.checkbox("Show The data"):
    st.write(df)
algorithm_choice = st.sidebar.selectbox("Algorithm you want to use:"
,('Random Forest','Support Victor Machine learning','Logistic Regression','Decision Tree','Key Neighbors'))
def parametrs_ui(algorithm):
    parametres = dict()
    if algorithm == "Random Forest":
        max_depth = st.sidebar.slider("Max depth",2,15)
        n_estimators = st.sidebar.slider('Number of Estimators',1,100)
        parametres["n_estimators"] = n_estimators
        parametres["max_depth"] = max_depth
    elif algorithm == "Support Victor Machine learning":
        C = st.sidebar.slider("C",0.01,10.0)
        parametres["C"] = C
    elif algorithm == "Logistic Regression":
        n_iters = st.sidebar.slider("Number of iters",100,1000)
        parametres["n_iters"] = n_iters
    elif algorithm == "Decision Tree":
        max_depth_ = st.sidebar.slider("Max depth",1,12)
        parametres["max_depth_"] = max_depth_
    else:
        n_neighbors = st.sidebar.slider("Number of neighbors",1,15)
        p = st.sidebar.slider("p",1,15)
        parametres["n_neighbors"] = n_neighbors
        parametres["p"] = p
    return parametres

def get_model(algorithme,parametres):
    if algorithme == "Random Forest":
        model = RandomForestClassifier(max_depth=parametres["max_depth"],n_estimators=parametres["n_estimators"])
    elif algorithme == "Support Victore Machine learning":
        model =SVC(C=parametres["C"])
    elif algorithme == "Logistic Regression":
        model = LogisticRegression(max_iter=parametres["n_iters"])
    elif algorithme == "Decision Tree":
        model = DecisionTreeClassifier(max_depth=parametres["max_depth_"])
    
    else:
        model = KNeighborsClassifier(n_neighbors=parametres["n_neighbors"],p=parametres["p"])
    return model

parametres= parametrs_ui(algorithm_choice)

model = get_model(algorithm_choice,parametres)

x = df[['age','creatinine_phosphokinase','serum_creatinine','high_blood_pressure','diabetes','smoking']]
y = df['DEATH_EVENT']
model.fit(x,y)
age = st.number_input("Age")
creatinine_phosphokinase = st.slider("Creatinine Phosphokinase",1,10000)
serum_creatinine = st.slider("Serum creatinine",0.0,10.0)
high_blood_pressure = st.selectbox("High blood pressure",("Yes","No"))
diabetes = st.selectbox("Diabetes",("Yes","No"))
smoking = st.selectbox("Smoking",("Yes","No"))

if st.button("Show Prediction"):
    
    x__ = [[age,creatinine_phosphokinase,serum_creatinine,1 if high_blood_pressure=='Yes' else 0,1 if diabetes == "Yes" else 0,1 if smoking=="Yes" else 0]]
    y_predcit_user ="Yes" if model.predict(x__) ==1 else "No"
    st.write(f"**Death event :{y_predcit_user}**")
y_predcit = model.predict(x)

st.subheader(f"The acuaracy :(`{accuracy_score(y_predcit,y)}`)")
all_columns = df.columns.tolist()
if st.checkbox("Show Columns"):
    name_columns = st.multiselect("Name of columns",all_columns)
    st.write(df[name_columns])
st.set_option('deprecation.showPyplotGlobalUse', False)
if st.sidebar.checkbox("Exploration of Data"):
    st.write("""
## ***Exploration***:
 Those some option ot ploting you data :
""")
    
    heat_map = st.checkbox("Heat Map")
    

    # Heat map
    if heat_map:
        st.text("You can choice the columns that you want to plot it in sid bar")
        selected_choice = st.sidebar.multiselect("Select the Columns `heat map`",all_columns)
        if selected_choice == []:
            sns.heatmap(df.corr())
        else:
            sns.heatmap(df[selected_choice].corr())
        st.pyplot()
    
  
    # Ploting
    ploting = st.checkbox("Plot") 
    if ploting:
        st.text("You can choice the columns that you want to plot it in sid bar")
        selected_choice = st.sidebar.multiselect("Choice The Columns `plot`",all_columns)
        if selected_choice == []:
            plt.plot()
        else :
            df[selected_choice].plot(figsize=(12,7))
        st.pyplot()
    # Bar
    bar = st.checkbox("Bar")
    if bar:
        st.text("You can choice the columns that you want to plot it in sid bar")
        st.info("The number 0 mean is Flase the number 1 is True(ex: sex-1 male,0 female)")
        selected_choice = st.sidebar.multiselect("Choice The Columns to `bar`",('DEATH_EVENT','sex','smoking','diabetes'))
        if selected_choice == []:
            plt.plot()
        else :
            df[selected_choice].value_counts().plot(kind='bar',figsize=(12,7))
            
        st.pyplot()


    #Pie
    pie = st.checkbox("Pie")
    if pie:
        st.text("You can choice the columns that you want to plot it in sid bar")
        st.info("The number 0 mean is Flase the number 1 is True(ex: sex-1 male,0 female)")
        selected_choice = st.sidebar.multiselect("Choice The Columns to `pie`",('DEATH_EVENT','sex','smoking','diabetes'))
        if selected_choice == []:
            plt.plot()
        else :
            df[selected_choice].value_counts().plot(kind='pie',figsize=(12,7))
            plt.title(selected_choice)
            
        st.pyplot()
    





