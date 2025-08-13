import gradio as gr
import pandas as pd
import random
import joblib
from sklearn import model_selection
import xgboost as xgb


THRESHOLD = 0.3
#loading model , scaler,and the data.
modelRF = joblib.load("fraud_model_RF.pkl")
modelXG = joblib.load("fraud_model.pkl")
modelCat = joblib.load("fraud_model_Cat.pkl")
scaler = joblib.load("scaler.pkl")
df = pd.read_csv("creditcard_small.csv") 
fraudulent_entries = df[df['Class'] == 1]
fraudulent_entries = fraudulent_entries.drop(columns=["Class"], errors="ignore")
df_features = df.drop(columns=["Class"], errors="ignore")
feature_names = df_features.columns.tolist()
#print(feature_names)



def random_fill():
    #print(fraudulent_entries.sample(1).iloc[0])
    #iloc takes only the data in the row
    row = fraudulent_entries.sample(1).iloc[0].tolist()
    return row

def random_fillNf():
    row = df_features.sample(1).iloc[0].tolist()
    return row

def predict(model_choice,Time, V1, V2, V3, V4, V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28,Amount):
    features = [[Time, V1, V2, V3, V4, V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28,Amount]]
    print(f"")
    print(f"[DEBUG] model choice: {model_choice}")
    #takin input data and scalling and adding features names to it.
    features_df = pd.DataFrame(features, columns=feature_names)
    features_scaled = scaler.transform(features_df)

    #choosing model depending on user input
    if model_choice == "CatBoost" :
        model = modelCat
    elif model_choice == "RandomF(bestRecall)" :
        model = modelRF
    else :    
        model = modelXG

    # predicting based on model choice.
    if model_choice == "CatBoost" :
        #model .predict gives an array [0] to get first element
        pred = model.predict(features_scaled)[0]

    elif model_choice == "RandomF(bestRecall)":
        pred = model.predict(features_scaled)[0]
    else:
        #XG requires data to be a dmatrix
        #print(f"[DEBUG] Received name: {features_scaled}")
        dmatrix = xgb.DMatrix(features_scaled)
        #print('test',features_scaled)
        preds = model.predict(dmatrix)
        pred = (preds > THRESHOLD).astype(int)
        print(f"pred: {pred}")

    return "Fraud" if pred == 1 else "Not Fraud"

with gr.Blocks() as demo:
    gr.Markdown("# Fraud Detection Project")
    # making 30 input bar.
    inputs = [gr.Number(label=name) for name in feature_names]
    model_choosen = gr.Dropdown(["CatBoost", "XG[Better]","RandomF(bestRecall)"], label="Select Model", value="XG[Better]")
    #randomize buttons
    fill_btnF = gr.Button("fill with a fraud")
    fill_btnNF = gr.Button("fill with a Non-fraud")
    #linking buttons with functions
    predict_btn = gr.Button("Predict")
    output = gr.Textbox(label="Prediction")
    fill_btnF.click(fn=random_fill, inputs=None, outputs=inputs)
    fill_btnNF.click(fn=random_fillNf, inputs=None, outputs=inputs)
    predict_btn.click(fn=predict, inputs=[model_choosen ]+  inputs  , outputs=output)

demo.launch(server_name="0.0.0.0", server_port=7860)

