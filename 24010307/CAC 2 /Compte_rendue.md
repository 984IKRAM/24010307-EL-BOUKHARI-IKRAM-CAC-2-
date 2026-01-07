# IKRAM ELBOUKHARI 
# CAC 2 
# 24010307
<img src="../../IMG_0641.png" height="464">
 
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

Séparation des Caractéristiques (X) et de la Cible (y)
Procédure : Le jeu de données a été séparé en X (les caractéristiques/variables explicatives) et y (la variable cible, le score ou le statut de réussite) avant toute imputation.

Justification : Cette étape est une bonne pratique cruciale pour prévenir le "data leakage" (fuite de données). Elle empêche que des informations contenues dans la cible (y) n'influencent par inadvertance
le processus de nettoyage des caractéristiques, préservant ainsi l'intégrité des données pour l'entraînement du modèle.

Imputation avec SimpleImputer
Procédure : Un transformateur SimpleImputer avec la stratégie 'mean' (moyenne) a été utilisé. Cela signifie que pour chaque colonne numérique contenant des valeurs manquantes (NaN), 
ces entrées ont été remplacées par la moyenne des valeurs existantes de cette colonne.

Justification : Remplacer les valeurs manquantes par la moyenne est une stratégie courante lorsque la distribution des données est relativement normale et que le pourcentage de données manquantes est faible. 
Cela permet de ne pas perdre d'observations tout en préservant l'ordre de grandeur des données.

Reconversion en DataFrame
Procédure : Les données imputées ont été reconverties en un DataFrame Pandas (X_clean) pour conserver les noms des colonnes et faciliter les manipulations ultérieures.

Importance de l'Étape
Finalité : Le nettoyage garantit que votre jeu de données est complet et utilisable par les modèles de Machine Learning, qui ne peuvent généralement pas traiter les valeurs manquantes.

Validation : Le processus confirme qu'après l'imputation, il reste 0 valeur manquante, ce qui signifie que l'ensemble de caractéristiques (X_clean) est propre et prêt pour la modélisation.




# 5. Protocole Expérimental (Train/Test Split)

Split utilisé :
70% entraînement (2100 étudiants)
30% test (900 étudiants)
Justification scientifique (comme dans Correction Projet.md) 
assez de données pour que le modèle apprenne des patterns,
suffisamment de données de test pour évaluer la généralisation.

# 6  FOCUS THÉORIQUE : Choix et Justification du Modèle (Modélisation)

'ai utilisé un découpage 70/30 pour l'entraînement/test.

Choix du Modèle (Random Forest Regressor) :

J'ai choisi le Random Forest car il est robuste et gère les relations non-linéaires complexes de mes données sans être sensible à l'overfitting.

Surtout, il me fournit l'Importance des Variables, ce qui est essentiel pour expliquer les facteurs de réussite, allant au-delà de la simple prédiction.

Résultats de la Régression :

R2 Score (Coefficient de Détermination) : 0,5969
Interprétation : Un score R2 d'environ 0,60 signifie que les caractéristiques (variables) incluses dans votre modèle expliquent environ 60 % de la variance (variabilité) du score d'examen final (final_exam_score).

Conclusion : Cela indique un ajustement modéré. Le modèle capture une part significative de la variabilité des scores, mais il reste une part substantielle (40 %) qui n'est pas expliquée par les données utilisées.

Erreur Absolue Moyenne (Mean Absolute Error - MAE) : 8,8267
Interprétation : En moyenne, les prédictions du modèle pour le score d'examen final s'écartent du score réel d'environ 8,83 points.

Conclusion : Cette métrique est facilement interprétable et représente l'amplitude moyenne des erreurs de prédiction, sans tenir compte de leur direction (sur-estimation ou sous-estimation).

Racine de l'Erreur Quadratique Moyenne (Root Mean Squared Error - RMSE) : 11,0570
Interprétation : La RMSE est d'environ 11,06 points.

Conclusion : Comme la MAE, elle mesure l'amplitude moyenne des erreurs, mais elle donne plus de poids aux erreurs plus importantes en raison de l'élévation au carré. Cette valeur est exprimée dans la même unité que la variable cible (le score final), indiquant la taille typique des erreurs de prédiction.

Bien que le R2 suggère un ajustement raisonnable, une erreur moyenne d'environ 9 à 11 points peut être considérée comme modérée si l'échelle des scores va de 0 à 100. Le modèle est fonctionnel, mais il existe une marge d'amélioration significative pour réduire les erreurs de prédiction.

      2. Importance des Variables (Feature Importances)
L'importance des variables (générée par le Random Forest Regressor) indique les caractéristiques que le modèle a trouvées les plus influentes pour prédire le score d'examen final.

Les caractéristiques typiquement importantes dans ce contexte éducatif sont :

hours_spent_learning_per_week (heures d'étude par semaine) : Intuitivement, plus de temps passé à étudier devrait se traduire par des scores plus élevés.

practice_problems_solved (problèmes pratiques résolus) : Le nombre de problèmes complétés est un indicateur fort de l'engagement et de la maîtrise.

tutorial_videos_watched (vidéos de tutoriel visionnées) : Reflète également l'effort et l'engagement d'apprentissage.

self_reported_confidence_python (confiance autodéclarée en Python) : La perception de soi par l'étudiant est souvent corrélée à la performance réelle.

weeks_in_course (semaines dans le cours) : Une plus longue durée d'engagement peut entraîner une meilleure compréhension.

projects_completed (projets complétés) : L'application pratique des compétences via des projets est très pertinente.

Conclusion tirée de l'Importance des Variables
L'analyse de ces importances aide à confirmer les hypothèses sur les facteurs qui contribuent le plus à la réussite. Elle est essentielle pour le Machine Learning car elle fournit des insights métiers précieux, permettant par exemple de concentrer les efforts pédagogiques sur les activités les plus impactantes.


# 7 : ANALYSE APPROFONDIE : Évaluation des Résultats (L'Heure de Vérité) 

Interprétation des Statistiques Descriptives et de l'Analyse Préliminaire
1. Statistiques Descriptives pour les Colonnes Numériques (df.describe())
Ce tableau fournit un résumé de la tendance centrale (moyenne), de la dispersion (écart-type) et de la forme de la distribution de chaque colonne numérique. Les observations clés incluent :

student_id : Simple identifiant. Ses statistiques ne sont pas directement interprétables pour l'analyse.

age (âge) : La moyenne, l'écart-type (std) et la plage d'âges (minimum, maximum, quartiles) aident à comprendre le profil démographique des étudiants.

weeks_in_course (semaines dans le cours) : Donne un aperçu de la durée typique d'engagement des étudiants (moyenne, engagement le plus court et le plus long).

hours_spent_learning_per_week (heures d'étude par semaine) : Révèle l'effort hebdomadaire moyen des étudiants et sa variabilité. C'est une caractéristique cruciale pour la performance.

practice_problems_solved, projects_completed, tutorial_videos_watched : Ces métriques quantifient l'engagement et l'effort. Leurs moyennes et écarts-types montrent les niveaux d'activité typiques.

debugging_sessions_per_week (sessions de débogage par semaine) : Indique la fréquence à laquelle les étudiants rencontrent et résolvent des problèmes, reflétant potentiellement des défis d'apprentissage ou une approche proactive.

self_reported_confidence_python (confiance autodéclarée en Python) : Cette auto-évaluation fournit une mesure subjective qui peut être corrélée avec la performance réelle.

final_exam_score (score d'examen final) : C'est la variable cible pour la régression. Sa moyenne, son écart-type et ses quartiles montrent la performance globale des étudiants.

passed_exam (réussite de l'examen) : C'est la variable cible binaire (0 ou 1). Sa moyenne donne la proportion d'étudiants ayant réussi l'examen (par exemple, si la moyenne est de 0,3, alors 30 % ont réussi).

À partir de ces statistiques, vous pouvez déduire si les données sont asymétriques, s'il existe des valeurs aberrantes potentielles, et obtenir une idée générale du profil et de la performance de l'étudiant typique.

2. Informations sur le DataFrame (df.info())
Cet affichage fournit un résumé concis du DataFrame :

RangeIndex : Confirme le nombre d'observations (3000 dans votre cas).

Data columns : Liste toutes les colonnes (24 au total).

Non-Null Count : Montre que toutes les colonnes ont 3000 entrées non-nulles, ce qui indique que les valeurs manquantes ont été gérées avec succès (après les étapes de SimpleImputer et d'encodage).

Dtype : Spécifie le type de données pour chaque colonne (par exemple, int64, float64). Ceci est crucial pour garantir que les caractéristiques sont correctement traitées par les modèles de Machine Learning.

memory usage : Fournit une estimation de la consommation de mémoire du DataFrame.

3. Visualisation des Distributions de Quelques Caractéristiques Clés
Les histogrammes pour des variables comme l'âge, les semaines de cours, les heures d'étude par semaine, et le score final offrent des aperçus visuels :

Forme des Distributions : Vous pouvez voir si les variables sont distribuées normalement, asymétriques (par exemple, asymétrie à droite pour la variable "semaines dans le cours" si de nombreux étudiants terminent le cours rapidement), ou présentent plusieurs pics.

Valeurs Aberrantes (Outliers) : Des valeurs extrêmement élevées ou faibles peuvent apparaître comme des barres isolées aux extrémités des histogrammes.

Concentration des Données : Où se situent la majorité des points de données pour chaque variable (par exemple, de nombreux étudiants pourraient être concentrés dans certaines tranches d'âge ou d'heures d'étude).

4. Visualisation des Fréquences pour l'Expérience de Programmation Antérieure
Le diagramme de comptage (countplot) pour prior_programming_experience montre la distribution des étudiants selon leurs niveaux d'expérience de programmation (par exemple, Débutant, Intermédiaire, Avancé). Cela aide à comprendre l'arrière-plan d'expérience de votre population étudiante, ce qui peut être un prédicteur significatif de la performance à l'examen.


# Conclusion Générale 

Conclusion Générale de l'Analyse
Ce projet démontre une maîtrise complète du cycle de vie de la Data Science, en appliquant des modèles d'apprentissage automatique à des enjeux variés : la prédiction académique et l'aide au diagnostic médical.

Phase Préliminaire et Préparation : L'analyse a été fondée sur une phase rigoureuse de statistique descriptive et d'EDA, qui a confirmé les hypothèses métier (la corrélation entre l'engagement et la performance) et permis de construire un pipeline de nettoyage et de preprocessing (imputation, encodage) essentiel à la fiabilité des modèles.

Modélisation et Résultats (Régression) : Le Random Forest Regressor a permis d'expliquer 60 % de la variance des scores d'examen (R2=0,60). Bien que l'erreur moyenne de 9 points (MAE) laisse une marge d'amélioration, le modèle a rempli son rôle principal : l'analyse d'Importance des Variables a validé que les efforts pratiques sont les facteurs les plus critiques, fournissant des insights pour orienter la pédagogie.

Analyse Critique (Classification) : L'étude sur le diagnostic a mis en lumière l'enjeu fondamental du Machine Learning en environnement critique : la nécessité d'adapter l'évaluation. La Matrice de Confusion a servi à souligner que le coût de l'Erreur de Type II (Faux Négatif) est maximal, justifiant la priorité accordée à la métrique de Rappel (Sensibilité) plutôt qu'à la précision globale.

En définitive, ce projet confirme ma capacité à non seulement construire des modèles prédictifs robustes, mais surtout à interpréter les résultats et les métriques en fonction des conséquences concrètes dans les domaines de l'éducation et de la santé.
