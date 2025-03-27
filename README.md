# Classification de Fraude à la Carte Bancaire

Ce projet est une application Streamlit permettant de comparer différents algorithmes de machine learning pour détecter les fraudes à la carte bancaire. L'objectif est de fournir un outil interactif pour visualiser les performances des modèles et comprendre les facteurs influençant les prédictions.

## Fonctionnalités

- **Comparaison de modèles** : Logistic Regression, Random Forest, etc.
- **Visualisation des métriques** : Matrice de confusion, scores de précision, rappel, F1.
- **Interface intuitive** : Créée avec Streamlit pour une utilisation simple et interactive.
- **Gestion du déséquilibre des classes** : Techniques d'échantillonnage pour équilibrer les données.


## Prérequis

- Python 3.8 ou plus
- Les bibliothèques suivantes :
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - streamlit

## Installation

1. Clonez ce dépôt :
   ```bash
   git clone https://github.com/votre-utilisateur/nom-du-repo.git
   cd nom-du-repo
   ```

2. Créez un environnement virtuel :
   ```bash
   python -m venv env
   source env/bin/activate  # Sur Windows : env\Scripts\activate
   ```

3. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## Utilisation

1. Lancez l'application Streamlit :
   ```bash
   streamlit run main.py
   ```

2. Ouvrez votre navigateur et accédez à l'adresse indiquée (généralement `http://localhost:8501`).


## Contribution

Les contributions sont les bienvenues ! Suivez ces étapes pour contribuer :
1. Forkez ce dépôt.
2. Créez une branche pour votre fonctionnalité (`git checkout -b feature-nom-fonctionnalite`).
3. Commitez vos modifications (`git commit -m 'Ajout d'une fonctionnalité'`).
4. Poussez votre branche (`git push origin feature-nom-fonctionnalite`).
5. Ouvrez une Pull Request.

## Auteurs

- **Robin Morelon** - Développeur principal


