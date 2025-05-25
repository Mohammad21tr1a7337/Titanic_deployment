import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# --- Page Setup ---
st.set_page_config(page_title="🚢 Titanic Survival Predictor", layout="centered")
st.title("🚢 Titanic Survival Predictor")
st.markdown("### Would you have survived the Titanic disaster?")
st.markdown("Use this simulation based on a **logistic regression model** trained on Titanic data to predict your chances of survival.")

# --- Sidebar Input ---
st.sidebar.header("🧾 Input Passenger Information")

def user_input_feature():
    age = st.sidebar.slider("Age", 1, 80, 25)
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
    embarked = st.sidebar.selectbox("Port of Embarkation", ["Cherbourg", "Southampton", "Queenstown"])
    siblings = st.sidebar.slider("Number of Siblings/Spouses Aboard", 0, 5, 0)
    parents = st.sidebar.slider("Number of Parents/Children Aboard", 0, 5, 0)

    gender_code = 1 if gender == "Male" else 0
    embarked_map = {"Cherbourg": 1, "Southampton": 2, "Queenstown": 3}
    embarked_code = embarked_map[embarked]

    data = {
        "Age": age,
        "Sex": gender_code,
        "Pclass": pclass,
        "Embarked": embarked_code,
        "SibSp": siblings,
        "Parch": parents
    }

    return pd.DataFrame(data, index=[0])

user_data = user_input_feature()
st.markdown("#### 👤 Passenger Input Summary")
st.dataframe(user_data)

# --- Load and Preprocess Data ---
df = pd.read_csv("Titanic_train.csv")
df.dropna(subset=['Age', 'Embarked'], inplace=True)

# Label Encoding
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])

# Scaling
scaler = MinMaxScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# Features and Target
features = ['Age', 'Sex', 'Pclass', 'SibSp', 'Parch']
X = df[features]
y = df['Survived']

# Model
model = LogisticRegression()
model.fit(X, y)

# Predict
prediction = model.predict(user_data[features])[0]
probability = model.predict_proba(user_data[features])[0]

# --- Output ---
st.markdown("---")
st.subheader("🧠 Prediction Result")

if prediction == 1:
    st.success(f"🎉 You would have survived! (Probability: {probability[1]:.2f})")
else:
    st.error(f"💀 Sorry, you would not have survived. (Probability: {probability[1]:.2f})")

st.markdown("#### 🔍 Probability Breakdown")
st.write({
    "Not Survived": f"{probability[0]:.2f}",
    "Survived": f"{probability[1]:.2f}"
})

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit & scikit-learn | Titanic Dataset")
