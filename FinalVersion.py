import gradio as gr
import pandas as pd
import random
import joblib
from sklearn import model_selection
import xgboost as xgb

THRESHOLD = 0.3
modelRF = joblib.load("fraud_model_RF.pkl")
modelXG = joblib.load("fraud_model.pkl")
modelCat = joblib.load("fraud_model_Cat.pkl")
scaler = joblib.load("scaler.pkl")
df = pd.read_csv("/Users/abdullahyehia/Desktop/data/creditcard.csv")  # already downloaded
fraudulent_entries = df[df['Class'] == 1]
fraudulent_entries = fraudulent_entries.drop(columns=["Class"], errors="ignore")
df_features = df.drop(columns=["Class"], errors="ignore")


feature_names = df_features.columns.tolist()
#print(feature_names)


def random_fill():
    row = fraudulent_entries.sample(1).iloc[0].tolist()
    return row

def predict(model_choice,Time, V1, V2, V3, V4, V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28,Amount):
    features = [[Time, V1, V2, V3, V4, V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28,Amount]]
    print(f"")
    print(f"[DEBUG] model choice: {model_choice}")
    features_df = pd.DataFrame(features, columns=feature_names)
    features_scaled = scaler.transform(features_df)
    #model = modelCat if model_choice == "CatBoost" else modelXG

    if model_choice == "CatBoost" :
        model = modelCat
    elif model_choice == "RandomF(bestRecall)" :
        model = modelRF
    else :    
        model = modelXG


    if model_choice == "CatBoost" :
        pred = model.predict(features_scaled)[0]

    elif model_choice == "RandomF(bestRecall)":
        pred = model.predict(features_scaled)[0]
    else:
        #print(f"[DEBUG] Received name: {features_scaled}")
        dmatrix = xgb.DMatrix(features_scaled)
        #print('test',features_scaled)
        preds = model.predict(dmatrix)
        pred = (preds > THRESHOLD).astype(int)
        print(f"pred: {pred}")


    #pred = model.predict(features_scaled)[0]
    return "Fraud" if pred == 1 else "Not Fraud"

with gr.Blocks() as demo:
    gr.Markdown("# Fraud Detection Project")
    inputs = [gr.Number(label=name) for name in feature_names]
    model_choosen = gr.Dropdown(["CatBoost", "XG[Better]","RandomF(bestRecall)"], label="Select Model", value="RF[BetterF1]")
    fill_btn = gr.Button("🎲")
    predict_btn = gr.Button("Predict")
    #inputs.append(gr.Dropdown(["RF[BetterF1]", "XG[HighPrecision]"], label="Select Model", value="RF[BetterF1]"))
    output = gr.Textbox(label="Prediction")
    fill_btn.click(fn=random_fill, inputs=None, outputs=inputs)
    predict_btn.click(fn=predict, inputs=[model_choosen ]+  inputs  , outputs=output)

demo.launch(server_name="0.0.0.0", server_port=7860)

