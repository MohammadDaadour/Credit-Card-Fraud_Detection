import gradio as gr
import joblib
import xgboost as xgb

THRESHOLD = 0.6

# Load trained model & scaler
modelRF = joblib.load("fraud_model_RF.pkl")
modelXG = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
# "Time","V1","V2","V3","V4","V5","V6","V7","V8","V9"
# ,"V10","V11","V12","V13","V14","V15","V16","V17","V18",
# "V19","V20","V21","V22","V23","V24","V25","V26","V27","V28",
# "Amount"

def predict(model_choice,Time, V1, V2, V3, V4, V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28,Amount):
    # Format input as a list of lists (1 row)
    features = [[Time, V1, V2, V3, V4, V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28,Amount]]
    print(f"")
    print(f"[DEBUG] model choice: {model_choice}")

    # Scale input (must match training pipeline)
    features_scaled = scaler.transform(features)
    model = modelRF if model_choice == "RF[BetterF1]" else modelXG
    # Get prediction
    if model_choice == "RF[BetterF1]" :
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

# Create Gradio interface
inputs = [
    gr.Dropdown(["RF[BetterF1]", "XG[betteRecall]"], label="Select Model", value="RF[BetterF1]"),
    gr.Number(label="Time"),
    gr.Number(label="Feature 1"),
    gr.Number(label="Feature 2"),
    gr.Number(label="Feature 3"),
    gr.Number(label="Feature 4"),
    gr.Number(label="Feature 5"),
    gr.Number(label="Feature 6"),
    gr.Number(label="Feature 7"),
    gr.Number(label="Feature 8"),
    gr.Number(label="Feature 9"),
    gr.Number(label="Feature 10"),
    gr.Number(label="Feature 11"),
    gr.Number(label="Feature 12"),
    gr.Number(label="Feature 13"),
    gr.Number(label="Feature 14"),
    gr.Number(label="Feature 15"),
    gr.Number(label="Feature 16"),
    gr.Number(label="Feature 17"),
    gr.Number(label="Feature 18"),
    gr.Number(label="Feature 19"),
    gr.Number(label="Feature 20"),
    gr.Number(label="Feature 21"),
    gr.Number(label="Feature 22"),
    gr.Number(label="Feature 23"),
    gr.Number(label="Feature 24"),
    gr.Number(label="Feature 25"),
    gr.Number(label="Feature 26"),
    gr.Number(label="Feature 27"),
    gr.Number(label="Feature 28"),
    gr.Number(label="Amount")
]

gr.Interface(fn=predict, inputs=inputs, outputs="text").launch()
