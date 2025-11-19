import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="고객 이탈률 예측", page_icon="📉")

# 1) 저장된 모델 로드
model = joblib.load("churn_model_2.pkl")

st.title("📉 신용카드 고객 이탈률 예측")
st.write("고객 정보를 입력하면 이탈률을 예측합니다")
st.divider()

# 2) 입력 폼 구성
st.subheader("📌 고객 기본 정보 입력")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("나이", min_value=18, max_value=100, value=40)
    marital = st.selectbox("결혼 여부", ["미혼", "기혼", "이혼", "미상"])
    edu = st.selectbox(
        "학력 수준",
        ["무학", "고등학교 졸업", "대학교 재학/졸업", "대학원 졸업", "박사 과정", "박사 학위", "미상"]
    )

with col2:
    gender = st.selectbox("성별", ["남자", "여자"])
    dependent = st.number_input("부양가족 수", min_value=0, max_value=10, value=1)
    income = st.selectbox(
        "소득 수준",
        ["$40K 미만","$40K - $60K","$60K - $80K","$80K - $120K","$120K 이상","미상"]
    )

st.divider()
# 3) 행동 정보 입력
st.subheader("📌 고객 행동 데이터 입력")

col3, col4 = st.columns(2)

with col3:
    total_ct = st.number_input("최근 거래건수", min_value=1, max_value=300, value=50)
    total_amt = st.number_input("최근 거래금액", min_value=1, max_value=50000, value=2000)
    ct_chng = st.number_input("거래건수 변화율", min_value=0.0, max_value=3.5, value=1.0)
    amt_chng = st.number_input("결제금액 변화율", min_value=0.0, max_value=3.5, value=1.0)

with col4:
    util = st.number_input("카드 사용률", min_value=0.0, max_value=1.0, value=0.3)
    inact = st.number_input("최근 1년 비활성 개월수", min_value=0, max_value=12, value=3)
    contact = st.number_input("고객센터 접촉 횟수", min_value=0, max_value=15, value=1)


# 4) 파생변수 생성
amt_per_trans = total_amt / total_ct

st.text(" ")
# 5) 버튼 클릭 시 예측 수행
center1, center2, center3 = st.columns([1, 1, 1])
with center2:
    run_predict = st.button("🔍 이탈 위험도 예측하기")

if run_predict:

    # 입력데이터 DataFrame
    input_df = pd.DataFrame([{
        "Customer_Age": age,
        "Gender": gender,
        "Dependent_count": dependent,
        "Education_Level": edu,
        "Marital_Status": marital,
        "Income_Category": income,
        "Total_Trans_Ct": total_ct,
        "Total_Trans_Amt": total_amt,
        "Total_Ct_Chng_Q4_Q1": ct_chng,
        "Total_Amt_Chng_Q4_Q1": amt_chng,
        "Avg_Utilization_Ratio": util,
        "Months_Inactive_12_mon": inact,
        "Contacts_Count_12_mon": contact,
        "Amt_Per_Trans": amt_per_trans
    }])

    proba = model.predict_proba(input_df)[0,1]
    percent = round(proba * 100, 2)

    # 색상 선택
    if percent < 20:
        color = "green"
    elif percent < 50:
        color = "orange"
    else:
        color = "red"

    # HTML로 색상 적용
    st.markdown(
        f"<h3>📉 이탈 확률: <b><span style='color:{color};'>{percent}%</span></b></h3>",
        unsafe_allow_html=True
    )

    # 위험도 단계 표기
    if percent < 20:
        st.success("🟢 위험도 낮음 (Low Risk)")
    elif percent < 50:
        st.warning("🟡 중간 위험도 (Medium Risk)")
    else:
        st.error("🔴 위험도 높음 (High Risk)")
