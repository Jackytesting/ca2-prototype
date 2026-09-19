# High-score exemplar app — self-contained (re-loads data + re-trains on Streamlit Cloud's fresh server).
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. 網頁基礎設定（置頂，設定寬螢幕與標題）
st.set_page_config(
    page_title="CA2 Prototype - Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/Giskard-AI/examples/main/datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv"
RANDOM_STATE = 42

@st.cache_data
def load_and_train():
    df = pd.read_csv(DATA_URL)
    y = (df["Churn"] == "Yes").astype(int)
    contract_ohe = pd.get_dummies(df["Contract"], prefix="Contract")
    internet_ohe = pd.get_dummies(df["InternetService"], prefix="Internet")
    X = pd.concat([df[["tenure", "MonthlyCharges"]], contract_ohe, internet_ohe], axis=1)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=RANDOM_STATE)
    model.fit(Xtr, ytr)
    return df, model, list(X.columns)

# 載入資料與模型
df, model, feature_cols = load_and_train()

# 2. 標題與標頭設計
st.title("📊 CA2 Prototype — High Score Exemplar")
st.markdown("### 🔮 Telco Customer Churn Prediction Dashboard")
st.write("Predicts **Churn**. Backed by a real trained `RandomForestClassifier` (4 business features).")

# 將警告訊息轉化為高質感的提示框
st.info(
    "💡 **Model Evaluation Disclaimer:** "
    "Remember that this model's Recall is only ~46% — it misses over half of real churners. "
    "A higher overall accuracy does not automatically mean a 'good enough' model for business decisions.",
    icon="⚠️"
)

st.markdown("---")

# 3. 畫面左右分欄 (Layout Layout)
# 左邊 45% 放輸入控制項，右邊 55% 放即時預測結果
col1, col2 = st.columns([45, 55], gap="large")

with col1:
    st.subheader("⚙️ Customer Attributes")
    st.markdown("Adjust the variables below to simulate a customer profile:")
    
    # 用容器包起來讓介面更整齊
    with st.container(border=True):
        tenure = st.slider("Tenure (Months)", 0, 72, 12, help="Number of months the customer has stayed with the company")
        monthly = st.slider("Monthly Charges (\$)", 18.0, 119.0, 70.0, help="The amount charged to the customer monthly")
        
        # 將兩個下拉選單並排，省空間又美觀
        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            contract = st.selectbox("Contract Type", sorted(df["Contract"].unique().tolist()))
        with sub_col2:
            internet = st.selectbox("Internet Service", sorted(df["InternetService"].unique().tolist()))

# 4. 右側即時計算與視覺化結果
with col2:
    st.subheader("🎯 Prediction Output")
    st.markdown("Live analysis based on the current active attributes:")

    # 機器學習核心計算
    row = pd.DataFrame([{"tenure": tenure, "MonthlyCharges": monthly}])
    for c in feature_cols:
        if c.startswith("Contract_"):
            row[c] = 1 if c == f"Contract_{contract}" else 0
        elif c.startswith("Internet_"):
            row[c] = 1 if c == f"Internet_{internet}" else 0
    row = row[feature_cols]
    
    pred = model.predict(row)[0]
    proba = model.predict_proba(row)[0][1]
    
    # 根據預測結果決定顏色與標籤
    if pred == 1:
        status_label = "HIGH RISK OF CHURN"
        status_type = "error"  # 紅色
        metric_delta = "Action Required"
    else:
        status_label = "LOW RISK (LOYAL CUSTOMER)"
        status_type = "success"  # 綠色
        metric_delta = "Stable Profile"

    # 呈現結果卡片
    with st.container(border=True):
        # 顯示狀態
        if pred == 1:
            st.error(f"🚨 **Result:** {status_label}")
        else:
            st.success(f"✅ **Result:** {status_label}")
        
        # 運用 st.metric 做出漂亮的數字大卡片
        m1, m2 = st.columns(2)
        with m1:
            st.metric(
                label="Churn Probability", 
                value=f"{proba:.1%}", 
                delta=metric_delta, 
                delta_color="inverse" if pred == 1 else "normal"
            )
        with m2:
            # 進度條直觀展示機率
            st.write("**Risk Gauge:**")
            st.progress(proba)

        # 額外的商業建議
        st.markdown("**💡 Next Best Action Suggestions:**")
        if pred == 1:
            st.markdown("- [ ] Send a personalized discount offer for Contract extension.")
            st.markdown("- [ ] Tech support checkup on Internet service quality.")
        else:
            st.markdown("- [ ] Eligible for standard loyalty rewards program.")
            st.markdown("- [ ] Consider cross-selling premium add-ons.")
