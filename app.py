import streamlit as st
import pandas as pd

st.set_page_config(page_title="Nueva Vizcaya Dengue AI Dashboard", layout="wide")

comp = pd.read_csv("data/MODEL_COMPARISON_TABLE_allinone.csv")
test = pd.read_csv("data/COMPARE_Test_2022_Observed_allinone.csv", parse_dates=["DATE"])
fcst = pd.read_csv("data/FORECAST_2023_2025_Demo_allinone.csv", parse_dates=["DATE"])

st.title("AI-Driven Dengue Outbreak Prediction Dashboard (Nueva Vizcaya)")

st.subheader("Model Comparison (Test Year: 2022 Observed)")
st.dataframe(comp, use_container_width=True)

# Pick best model automatically (lowest MAE)
best_model = comp.sort_values("MAE_2022").iloc[0]["Model"]
st.write(f"**Best model on 2022 observed:** {best_model}")

# Map model names to columns
test_map = {
    "XGBoost": "XGB_PRED",
    "RandomForest": "RF_PRED",
    "LSTM": "LSTM_PRED",
    "Ensemble (weighted)": "ENSEMBLE_PRED",
    "Ensemble": "ENSEMBLE_PRED",
}
fcst_map = {
    "XGBoost": "XGB_PRED",
    "RandomForest": "RF_PRED",
    "LSTM": "LSTM_PRED",
    "Ensemble (weighted)": "ENSEMBLE_PRED",
    "Ensemble": "ENSEMBLE_PRED",
}

pred_col_test = test_map.get(best_model, "LSTM_PRED")
pred_col_fcst = fcst_map.get(best_model, "LSTM_PRED")

st.subheader("2022 Observed: Actual vs Predicted")
plot_df = test[["DATE", "ACTUAL_CASES", pred_col_test]].rename(columns={pred_col_test: "PREDICTED"})
st.line_chart(plot_df.set_index("DATE"))

st.subheader("Forecast Demo (2023–2025)")
plot_fc = fcst[["DATE", pred_col_fcst]].rename(columns={pred_col_fcst: "FORECASTED_CASES"})
st.line_chart(plot_fc.set_index("DATE"))

st.caption("Note: 2023–2025 are forecasted estimates only and are not evaluated against observed ground truth.")
