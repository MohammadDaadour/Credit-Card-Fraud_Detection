import streamlit as st
import pandas as pd
import random
import joblib
from sklearn import model_selection
import xgboost as xgb


#if "model" not in st.session_state:
st.session_state.model = joblib.load("fraud_model.pkl")
st.session_state.modelRf = joblib.load("fraud_model_RF.pkl")
st.session_state.modelCat = joblib.load("fraud_model_Cat.pkl")
st.session_state.scaler = joblib.load("scaler.pkl")

df = pd.read_csv("/Users/abdullahyehia/Desktop/data/creditcard.csv")  # already downloaded
fraudulent_entries = df[df['Class'] == 1]
fraudulent_entries = fraudulent_entries.drop(columns=["Class"], errors="ignore")
df_features = df.drop(columns=["Class"], errors="ignore")
feature_names = df_features.columns.tolist()

if "inputs" not in st.session_state:
    st.session_state.inputs = {f: None for f in feature_names}


st.title("Fraud detection app")
#st.write("Fraud detection app")

with st.sidebar :
    st.title("sources:" )
    url = "https://nilsonreport.com/articles/card-fraud-losses-worldwide-in-2023/#:~:text=Fraud%20losses%20incurred%20by%20card,losses%20were%20%2433.45%20billion%20worldwide."
    st.write("nilsonreport [link](%s)" % url)
    url = "https://www.clearlypayments.com/blog/credit-card-fraud-statistics-in-2024-for-usa/#:~:text=The%20United%20States%20consistently%20experiences,according%20to%20the%20Nilson%20Report."
    st.write("clearlypayments [link](%s)" % url)


    



st.header(" Credit Card Fraud Overview")

col1, col2, col3 = st.columns(3,gap='small',)

with col1:
    col1.metric("US Fraud Losses", "$12.5 billion ")
    st.caption("Reported to FTC in 2023 : US Fraud Losses (2023)")

with col2:
    col2.metric("Highest country in losses(2024)", "40%")
    st.caption(' of global card fraud losses in the US')

with col3:
    col3.metric("Global Card Fraud Losses (2023)", " $33.45")
    st.caption('billion worldwide')

left, middle= st.columns(2)

randomizbtn = left.button("Randomize with fraude", type="primary")
if randomizbtn:
    rand_row = fraudulent_entries.sample(1).iloc[0]
    for f in feature_names:
        st.session_state.inputs[f] = float(rand_row[f])

randomizNFbtn = middle.button("Randomize with non fraude", type="primary")

if randomizNFbtn:
    rand_row = df_features.sample(1).iloc[0]
    for f in feature_names:
        st.session_state.inputs[f] = float(rand_row[f])







with st.form("Data input"):
    st.write("Enter all the features")
    # Dropdown for model choice
    model_choice = st.selectbox(
        "Choose Model",
        options=["Catboost", "Random Forest", "XGBoost"]
    )
    for f in feature_names:
        st.session_state.inputs[f] = st.number_input(
            f"Enter {f}",
            key=f, 
            value=st.session_state.inputs[f] if st.session_state.inputs[f] is not None else 0.0
        )
    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        #st.write(f"Time: {st.session_state.Time}")
        inputs = []
        model_selected=model_choice
        st.write(f"Name: {model_choice}")
        for f in feature_names:
            extracted = st.session_state.inputs[f]
            inputs.append(extracted)
            #st.write(extracted)
        x= pd.DataFrame([inputs], columns=feature_names)
        x_scaled= st.session_state.scaler.transform(x)
        

        if model_choice == "CatBoost" :
            model = st.session_state.modelCat
        elif model_choice == "Random Forest" :
            model = st.session_state.modelRf
        else :    
            model = st.session_state.model


        if model_choice == "CatBoost" :
            pred = model.predict(x_scaled)[0]

        elif model_choice == "Random Forest":
            pred = model.predict(x_scaled)[0]
        else:
            #print(f"[DEBUG] Received name: {features_scaled}")
            dmatrix = xgb.DMatrix(x_scaled)
            #print('test',features_scaled)
            preds = model.predict(dmatrix)
            pred = (preds > 0.3).astype(int)
            print(f"pred: {pred}")
            #prediction = st.session_state.model.predict(x_scaled)
        if pred[0] == 0 :
            output='Not Fraud'
            color= "green"
        else:
            output='Fraud'
            color= "red"
        st.markdown(f"<p style='font-size:30px; color:{color}; text-align: center;'>Prediction: {output}</p>", unsafe_allow_html=True)
        #st.success(f"Prediction: {output}")


#st.session_state.model = joblib.load("fraud_model.pkl")
    #st.session_state.modelRf = joblib.load("fraud_model_RF.pkl")
    #st.session_state.modelCat
