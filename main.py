

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier


st.title("Application de Détection de Fraude Bancaire -- Robin Morelon")
st.markdown("""
Cette application vous permet d'entraîner un modèle de machine learning pour détecter les fraudes bancaires.
Vous pouvez :
- Choisir une méthode de transformation des données (NearMiss, SMOTE ou aucune transformation).
- Sélectionner un modèle (Régression Logistique ou Random Forest).
- Visualiser les performances du modèle avec différentes métriques et une matrice de confusion.
""")

df = pd.read_csv("creditcard.csv")
df.drop(["Time"], axis=1, inplace=True)
df.drop_duplicates(inplace=True)
sc = StandardScaler()
df["Amount"] = sc.fit_transform(pd.DataFrame(df["Amount"]))
X = df.drop(["Class"], axis=1)
y = df["Class"]

# Affichage des données avant transformations
if st.checkbox("Afficher un aperçu des données"):
    st.write(df.head())

if st.checkbox("Distribution des classes avant transformation"):
    fig, ax = plt.subplots()
    sns.countplot(x=y, palette="pastel", ax=ax)
    ax.set_title("Distribution des Fraudes vs Transactions Normales")
    st.pyplot(fig)


# Séparation des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sélection de la méthode de transformation
st.subheader("Transformation des données")
st.write(f"Taille du dataset avant transformation: {len(X)}")
transformation = st.selectbox("Choisir une méthode de transformation :", ["Aucune transformation", "NearMiss", "SMOTE"])
if transformation == "NearMiss":
    nmi = NearMiss()
    X_train, y_train = nmi.fit_resample(X, y)
    st.subheader("Distribution après transformation")
    fig, ax = plt.subplots()
    sns.countplot(x=y_train, palette="pastel", ax=ax)
    st.pyplot(fig)
elif transformation == "SMOTE":
    smote = SMOTE()
    X_train, y_train = smote.fit_resample(X, y)
    st.subheader("Distribution après transformation")
    fig, ax = plt.subplots()
    sns.countplot(x=y_train, palette="pastel", ax=ax)
    st.pyplot(fig)
else:
    X_train = X
    y_train = y

st.write(f"Taille du dataset après transformation: {len(X_train)}")



# Sélection du modèle
st.subheader("Choix du modèle")
model_choice = st.selectbox("Sélectionnez un modèle :", ["Logistic Regression", "Decision Tree Classifier","RandomForestClassifier"])
if model_choice == "Logistic Regression":
    model = LogisticRegression()
elif model_choice == "Decision Tree Classifier":
    model = DecisionTreeClassifier(max_depth=5)
else:
    model = RandomForestClassifier(n_estimators=50,max_depth=10,class_weight="balanced",n_jobs=-1)

with st.spinner("Entraînement du modèle en cours..."):
    model.fit(X_train, y_train)
    st.success("Entraînement terminé !")

y_pred = model.predict(X_test)

# Validation croisée sur l’ensemble des données
cv_scores = cross_val_score(model, X, y, cv=5, scoring="f1")
st.write(f"Score moyen (cross-validation) : {cv_scores.mean():.2f}")

# Affichage des métriques
st.subheader("Résultats")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
col2.metric("Précision", f"{precision_score(y_test, y_pred):.2f}")
col3.metric("Recall", f"{recall_score(y_test, y_pred):.2f}")
col4.metric("F1 Score", f"{f1_score(y_test, y_pred):.2f}")

# Explication des métriques
with st.expander("Explication des métriques"):
    st.write("""
    - **Accuracy** : Proportion de prédictions correctes sur l’ensemble des données.
    - **Précision** : Parmi les transactions détectées comme frauduleuses, combien le sont réellement ?
    - **Recall** : Parmi toutes les fraudes, combien ont été bien détectées ?
    - **F1 Score** : Moyenne harmonique entre la précision et le recall.
    """)

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
st.write("Matrice de confusion :")
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Prédiction", fontsize=12)
ax.set_ylabel("Réel", fontsize=12)
ax.set_title("Matrice de Confusion", fontsize=14)
st.pyplot(fig)


# Courbe AUC-ROC
y_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

st.subheader("Courbe ROC-AUC")
y_train_proba = model.predict_proba(X_train)[:, 1]
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
roc_auc_train = auc(fpr_train, tpr_train)

fig, ax = plt.subplots()
ax.plot(fpr_train, tpr_train, color="blue", lw=2, label=f"Train ROC curve (AUC = {roc_auc_train:.2f})")
ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"Test ROC curve (AUC = {roc_auc:.2f})")
ax.plot([0, 1], [0, 1], color="navy", linestyle="--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Receiver Operating Characteristic (ROC)")
ax.legend(loc="lower right")
st.pyplot(fig)
