import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder


def load_model():
    with open("/home/teddy/Music/Java/new.ipynb/stud_per3/student_per3.pkl","rb") as file:
        model,scaler,le = pickle.load(file)
    return model,scaler,le

def preprocessing_input_data(data,scaler,le):
    data["Extracurricular Activities"]=le.transform([data["Extracurricular Activities"]])[0]
    df = pd.DataFrame([data])
    df_transformed = scaler.transform(df)
    return df_transformed

def predict_data(data):
    model,scaler,le =load_model()
    processed_data = preprocessing_input_data(data,scaler,le)
    prediction = model.predict(processed_data)

    return prediction

def main():
    st.title("Student Performance Prediction")
    st.write("Enter your data")
    hs = st.number_input("Hours Studied",min_value=1,max_value=10,value=5)
    ps = st.number_input("Previous Score",min_value=40,max_value=100,value=70)
    eca = st.selectbox("Extra curiculum activity",["Yes","No"])
    sh = st.number_input("Sleep Hours",min_value=4,max_value=10,value=7)
    nqps = st.number_input("Number of question paper solved",min_value=0,max_value=10,value=5)


    if st.button("Predict your score"):
        user_data ={
            "Hours Studied":hs,
            "Previous Scores":ps,
            "Extracurricular Activities":eca,
            "Sleep Hours": sh,
            "Sample Question Papers Practiced":nqps
        }
        prediction = predict_data(user_data)
        st.success(f"Your prediction is {prediction}")

if __name__ == "__main__":
    main()