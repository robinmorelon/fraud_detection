

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix


st.title("Application de Détection de Fraude Bancaire -- Robin Morelon")

df = pd.read_csv("creditcard.csv")
df.drop(["Time"], axis=1, inplace=True)
df.drop_duplicates(inplace=True)
sc = StandardScaler()
df["Amount"] = sc.fit_transform(pd.DataFrame(df["Amount"]))
X = df.drop(["Class"], axis=1)
y = df["Class"]

# Sélection de la méthode de transformation
st.subheader("Transformation des données")
transformation = st.selectbox("Choisir une méthode de transformation :", ["Aucune transformation", "TomekLinks", "SMOTE"])
if transformation == "TomekLinks":
    tl = TomekLinks()
    X, y = tl.fit_resample(X, y)
elif transformation == "SMOTE":
    smote = SMOTE()
    X, y = smote.fit_resample(X, y)

# Séparation des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sélection du modèle
st.subheader("Choix du modèle")
model_choice = st.selectbox("Sélectionnez un modèle :", ["Logistic Regression", "RandomForestClassifier"])
if model_choice == "Logistic Regression":
    model = LogisticRegression()
else:
    model = RandomForestClassifier(n_estimators=50,n_jobs=-1)

with st.spinner("Entraînement du modèle en cours..."):
    model.fit(X_train, y_train)
    st.success("Entraînement terminé !")

y_pred = model.predict(X_test)

# Affichage des métriques
st.subheader("Résultats")
st.write(f"Accuracy : {accuracy_score(y_test, y_pred):.2f}")
st.write(f"Precision : {precision_score(y_test, y_pred):.2f}")
st.write(f"Recall : {recall_score(y_test, y_pred):.2f}")
st.write(f"F1 Score : {f1_score(y_test, y_pred):.2f}")

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
st.write("Matrice de confusion :")
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Prédiction", fontsize=12)
ax.set_ylabel("Réel", fontsize=12)
ax.set_title("Matrice de Confusion", fontsize=14)
st.pyplot(fig)

