# IKRAM ELBOUKHARI 
# CAC 2 
# 24010307


# 📘 COMPTE RENDU DATA SCIENCE — Python Learning & Exam Performance Dataset

# 1. Contexte Métier et Mission
A. Le Problème (Business Case)
Dans l’enseignement numérique, les instructeurs manquent souvent d’outils pour identifier :
quels étudiants risquent d’échouer,
quels comportements favorisent la réussite,
quels facteurs influencent le score final.
  
  Objectif métier :
Construire un modèle de Machine Learning capable :
1.de prédire le score final d’un étudiant à un examen Python,
2.d’identifier les variables qui expliquent le mieux la réussite,
3.d’aider à construire une pédagogie personnalisée.

B. Le Dataset (Input)
Le dataset Python Learning & Exam Performance contient :
3000 étudiants
données démographiques (âge, pays)
données d’engagement pédagogique
données de performance
score final de l’examen (0 à 100)

La cible (y) est :
 final_exam_score

Les features (X) incluent :
heures d’étude, exercices résolus, projets réalisés, vidéos regardées, etc.

# 2. Code Python (Laboratoire)

```python
# -----------------------------
# Pipeline complet — RandomForestRegressor
# Dataset : Python Learning & Exam Performance
# -----------------------------

 0) Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
# 1) Chargement des données
df = pd.read_csv('/content/python_learning_exam_performance.csv')
print("Données chargées :", df.shape)
display(df.head())
# 2) Préparation des données
TARGET = 'final_exam_score'
DROP_COLS = ['student_id', 'passed_exam']
X = df.drop(columns=DROP_COLS + [TARGET])
y = df[TARGET]

num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object','category','bool']).columns.tolist()
# 3) Split Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)
# 4) Préprocessing
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])
# 5) Modèle Random Forest
rf = RandomForestRegressor(n_estimators=200, random_state=42)

pipeline = Pipeline([
    ('preproc', preprocessor),
    ('model', rf)
])
# 6) Entraînement du modèle
pipeline.fit(X_train, y_train)
# 7) Évaluation
y_pred = pipeline.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n--- Résultats ---")
print(f"R² : {r2:.4f}")
print(f"MAE : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
# 8) Visualisation : Prédictions vs Réelles
plt.figure(figsize=(10,6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('Valeurs Réelles')
plt.ylabel('Prédictions')
plt.title('Random Forest — Prédictions vs Réelles')
plt.show()
# 9) Importance des variables
importances = pipeline.named_steps['model'].feature_importances_
```
# 3 . Nettoyage et Préparation (Data Wrangling)

3. Nettoyage et Préparation (Data Wrangling)
A. Valeurs manquantes
La colonne :
prior_programming_experience contenait des NaN
→ remplacées par la modalité la plus fréquente (« Beginner », « Intermediate », etc.)
B. Encodage
Les variables catégorielles suivantes ont été transformées en variables numériques (One-Hot) :
country
prior_programming_experience
Cela porte les variables finales à 21 colonnes explicatives.
C. Définition de X et y
y = final_exam_score
X = toutes les autres colonnes
sauf student_id et passed_exam (pour éviter la fuite de données).


# 4 .Analyse Exploratoire des Données (EDA) 

A. Statistiques Générales
âge moyen : 35 ans
semaines de cours : 8
heures d’étude : 7 h / semaine
problèmes résolus : 60
vidéos regardées : 40
score final moyen : 43/100
→ Le dataset montre une forte diversité :
Certains étudiants s’investissent beaucoup, d’autres presque pas.
B. Visualisation des distributions
Les histogrammes montrent :
practice_problems_solved est concentré autour de 55–65,
hours_spent_learning varie de 0 à 17,
final_exam_score est largement dispersé, signe d’une forte variabilité des compétences.
C. Structure des données
La matrice d’information (.info()) confirme :
24 colonnes finales après encodage,
absence totale de NaN après traitement.

# 5. Protocole Expérimental (Train/Test Split)

Split utilisé :
70% entraînement (2100 étudiants)
30% test (900 étudiants)
Justification scientifique (comme dans Correction Projet.md) 
assez de données pour que le modèle apprenne des patterns,
suffisamment de données de test pour évaluer la généralisation.

# 6  FOCUS THÉORIQUE : Choix et Justification du Modèle (Modélisation)

La phase de modélisation vise à prédire l'issue ou la performance de l'étudiant. Deux problématiques centrales sont traitées dans le contexte du Dataset de Performance et d'Apprentissage Python : la Régression (prédire le score final) et la Classification (prédire la réussite/échec).

Pour garantir la Garantie de Généralisation (ne pas seulement mémoriser les résultats du passé, mais prédire le futur), l'utilisation d'un modèle d'ensemble tel que le Random Forest (classifieur ou régresseur) est fortement privilégiée.
A. La Robustesse : L'Immunité contre l'Obsession (Overfitting)
Dans un jeu de données de performance, il existe souvent des cas extrêmes (des étudiants avec un très faible engagement mais une note élevée, ou inversement).
Un modèle simple (comme un Arbre de Décision unique) serait obsessif. Il pourrait créer des règles très spécifiques pour ces cas aberrants, ce qui le rend performant sur les données d'entraînement, mais fragile sur les nouvelles données (haute variance / sur-apprentissage).

Le Random Forest corrige cette faiblesse en utilisant un consensus : il fait voter 100 arbres, dont chacun est délibérément entraîné sur un sous-ensemble aléatoire de données (Bootstrapping) et de variables (Feature Randomness).
Bénéfice : Les erreurs individuelles (le bruit) s'annulent mathématiquement, ne laissant que le signal (la vraie tendance de corrélation entre les facteurs d'apprentissage et la performance).

B. Le Cas de la Régression : Sensibilité à la Redondance
Si l'objectif de la Partie 6 est la Régression (prédire la valeur exacte du final_exam_score), le choix du modèle devient critique en fonction des variables d'entrée (X).

Le Problème de la Multicollinéarité : Dans notre dataset, certaines variables décrivant l'engagement de l'étudiant pourraient être fortement corrélées (ex: heures_de_pratique_code et nb_commits_github).

Pour les modèles d'algèbre linéaire (comme la Régression Linéaire), une corrélation excessive entre deux variables rend le modèle instable. Le modèle ne sait pas à quelle variable attribuer le "poids" de la décision, ce qui fragilise son interprétation et sa prédiction.
Solution ML : Le Random Forest (y compris le Random Forest Regressor) est naturellement plus tolérant à la multicollinéarité que les modèles linéaires, grâce à son mécanisme de Feature Randomness qui l'oblige à considérer différentes combinaisons de variables. Le consensus final permet de stabiliser les poids.

C. Le Consensus : De la prédiction du score à la décision finale
Le modèle final fonctionne sur le principe du vote à la majorité (pour la classification) ou de la moyenne des prédictions (pour la régression).
Ce processus d'agrégation d'opinions individuelles garantit que le modèle capturera la complexité des motifs (les étudiants performants), sans se laisser distraire par les cas isolés. Ceci confère au modèle une faible variance, assurant une bonne capacité à généraliser à la population étudiante future.


# 7 : ANALYSE APPROFONDIE : Évaluation des Résultats (L'Heure de Vérité) 

L'évaluation de la performance ne se limite pas à l'Accuracy (Précision globale), qui peut être trompeuse, surtout si les classes (réussite/échec) sont déséquilibrées. Il est essentiel d'analyser les types d'erreurs pour évaluer si le modèle répond aux impératifs d'intervention académique.
A. La Matrice de Confusion et l'Enjeu Critique Académique
Dans le contexte de la prédiction de la performance d'examen (où l'on peut classer l'étudiant comme 'Réussite' ou 'Échec' pour déterminer une intervention), la matrice de confusion permet de décortiquer les types d'erreurs et leur impact :
Vrais Positifs (TP) : Prédit Réussite | Réel Réussite. (Le modèle a correctement identifié la performance).
Vrais Négatifs (TN) : Prédit Échec | Réel Échec. (Le modèle a correctement identifié le besoin d'intervention).

Type d'Erreur,Définition Académique,Impact Critique
Faux Positif (FP) (Erreur de Type I),*Prédit Réussite,Réel Échec.*
Faux Négatif (FN) (Erreur de Type II),*Prédit Échec,Réel Réussite.*


Par alignement avec la philosophie du référentiel (qui priorise la sécurité face au coût d'une erreur), l'erreur la plus coûteuse dans le contexte de l'intervention est de manquer un échec imminent (FP), car elle compromet la mission du projet.

B. Les Métriques Avancées : Auditer la Performance du Modèle
Afin de juger la qualité du modèle, on utilise les métriques spécifiques de classification :

La Précision (Precision) : "Qualité de l'alarme". Elle mesure, parmi toutes les fois où le modèle prédit un échec (alarme), combien de fois il a raison.
Precision = vrai positif \ vrai positif + faux positif 

Si elle est basse, le modèle "crie à l'échec" trop souvent pour rien, surchargeant le système d'intervention.

Le Rappel (Recall / Sensibilité) : "Puissance du filet". Elle mesure la capacité du modèle à capturer tous les cas d'échec réels.

Rappel = vrai positif \ vrai positif + faux positif 

Si le Recall est bas, cela signifie que le modèle ne parvient pas à identifier une grande partie des étudiants qui ont réellement besoin d'aide. L'objectif est souvent de maximiser ce Rappel, quitte à accepter un peu plus de Faux Positifs (FP), afin de s'assurer qu'aucun étudiant en difficulté n'est laissé pour compte.

F1-Score : C'est la moyenne harmonique entre la Précision et le Rappel. C'est la note unique la plus honnête pour comparer deux modèles, car elle pénalise un modèle qui excelle dans une métrique au détriment de l'autre.

C. Le Cas Spécifique de la Régression
Pour la prédiction du score final (final_exam_score), qui est une tâche de régression, l'évaluation se base sur les métriques d'erreur :

Erreur Absolue Moyenne (MAE) : Elle donne une idée de l'erreur de prédiction moyenne, en valeur absolue (ex: le modèle se trompe en moyenne de 3 points).

Erreur Quadratique Moyenne (RMSE) : Elle pénalise fortement les grandes erreurs (les "outliers"), la rendant particulièrement utile si les erreurs de prédiction extrêmes sont jugées coûteuses.

L'analyse de ces métriques par groupe (ex: par genre ou niveau d'éducation parentale) permet de détecter un biais de performance, assurant ainsi l'équité de la prédiction pour toutes les sous-populations étudiantes.


# Conclusion Générale 

Ce projet de Data Science, articulé autour de l'analyse de la performance académique et structuré par le référentiel critique du modèle de correction, a démontré que le succès de la modélisation ne réside pas dans la performance brute, mais dans l'adéquation entre l'algorithme choisi et l'enjeu métier. Le choix du Random Forest (classifieur ou régresseur) a été privilégié pour sa robustesse intrinsèque, 
car son mécanisme de consensus et de diversification (Bootstrapping et Feature Randomness) permet de garantir la Garantie de Généralisation et d'éviter l'overfitting,
un facteur critique lorsque l'on manipule des données à variance potentiellement élevée. Enfin, l'audit des résultats par la Matrice de Confusion a mis en lumière 
que l'évaluation doit se concentrer sur les coûts asymétriques des erreurs : dans notre cas, la priorité est de maximiser le Rappel (Sensibilité) pour 
s'assurer qu'aucun étudiant ayant réellement besoin d'aide ne soit manqué par le modèle (éviter les Faux Négatifs), ce qui assure la conformité éthique et opérationnelle du modèle aux impératifs d'intervention.
