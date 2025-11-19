import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import joblib

df = pd.read_csv("credit_card_churn.csv")
df['churn_flag'] = (df['Attrition_Flag'] == 'Attrited Customer').astype(int)

# 파생 변수 생성
df["Amt_Per_Trans"] = df["Total_Trans_Amt"] / df["Total_Trans_Ct"]
# df["Utilization_to_Limit"] = df["Avg_Utilization_Ratio"] * df["Credit_Limit"]

# 최종 feature selection
"""feature_cols = [
    'Customer_Age','Gender','Dependent_count','Education_Level','Marital_Status','Income_Category',
    'Total_Trans_Ct','Total_Trans_Amt','Total_Ct_Chng_Q4_Q1','Total_Amt_Chng_Q4_Q1',
    'Avg_Utilization_Ratio','Months_Inactive_12_mon','Contacts_Count_12_mon','Credit_Limit',
    'Amt_Per_Trans','Utilization_to_Limit'
]"""

feature_cols = [
    'Customer_Age','Gender','Dependent_count','Education_Level','Marital_Status','Income_Category',
    'Total_Trans_Ct','Total_Trans_Amt','Total_Ct_Chng_Q4_Q1','Total_Amt_Chng_Q4_Q1',
    'Avg_Utilization_Ratio','Months_Inactive_12_mon','Contacts_Count_12_mon','Amt_Per_Trans'
]

X = df[feature_cols]
y = df['churn_flag']

"""num_cols = [
    'Customer_Age','Dependent_count','Total_Trans_Ct','Total_Trans_Amt',
    'Total_Ct_Chng_Q4_Q1','Total_Amt_Chng_Q4_Q1','Avg_Utilization_Ratio',
    'Months_Inactive_12_mon','Contacts_Count_12_mon','Credit_Limit',
    'Amt_Per_Trans','Utilization_to_Limit'
]"""

num_cols = [
    'Customer_Age','Dependent_count','Total_Trans_Ct','Total_Trans_Amt',
    'Total_Ct_Chng_Q4_Q1','Total_Amt_Chng_Q4_Q1','Avg_Utilization_Ratio',
    'Months_Inactive_12_mon','Contacts_Count_12_mon','Amt_Per_Trans'
]

cat_cols = ['Gender','Education_Level','Marital_Status','Income_Category']

numeric_tf = StandardScaler()
categorical_tf = OneHotEncoder(handle_unknown='ignore')

preprocess = ColumnTransformer(
    transformers=[
        ('num', numeric_tf, num_cols),
        ('cat', categorical_tf, cat_cols)
    ]
)

model = Pipeline(steps=[
    ('preprocess', preprocess),
    ('clf', RandomForestClassifier(n_estimators=300, random_state=42))
])

# Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("모델 학습 중...")
model.fit(X_train, y_train)
print("학습 완료!")

# 평가
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]

print("\n======== 모델 성능 평가 ========")
print("정확도:", accuracy_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_proba))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("혼동행렬:")
print(confusion_matrix(y_test, y_pred))

# 모델 저장
# joblib.dump(model, "churn_model.pkl")
joblib.dump(model, "churn_model_2.pkl")
print("\n모델 저장 완료: churn_model.pkl")
