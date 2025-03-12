import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# 🔹 1. Charger le dataset
df = pd.read_csv(r"C:\Users\Louay\Downloads\large_web_ad_attacks_dataset.csv")  # Remplace par ton chemin

# 🔹 2. Gérer les valeurs manquantes
df.fillna("", inplace=True)  # Remplace les NaN par des chaînes vides

# 🔹 3. Encoder la cible (Attack Type)
label_encoder = LabelEncoder()
df["Attack Type"] = label_encoder.fit_transform(df["Attack Type"])

# 🔹 4. Transformer les colonnes textuelles en vecteurs TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=300)  # Sélectionne les 300 mots les plus fréquents
X_text = tfidf_vectorizer.fit_transform(df["Payload"] + " " + df["URL"] + " " + df["Endpoint"]).toarray()

# 🔹 5. Sélectionner les caractéristiques utiles
X = pd.concat([df[["Status_Code", "Response_Size"]], pd.DataFrame(X_text)], axis=1)
y = df["Attack Type"]

# 🔹 6. Séparer les données en train et test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 7. Entraîner un modèle Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 🔹 8. Évaluer le modèle
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 🔹 9. Prédire une nouvelle attaque
nouvelle_donnee = pd.DataFrame({
    "Status_Code": [500],  # Erreur interne souvent causée par une injection SQL
    "Response_Size": [4000],  # Réponse volumineuse contenant potentiellement une erreur SQL
    "Payload": ["' OR '1'='1' --"],  # Payload classique d'injection SQL
    "URL": ["/login.php"],  # URL typique d'un point d'authentification vulnérable
    "Endpoint": ["/login.php"]
})


# Transformer la nouvelle donnée en vecteur TF-IDF
X_nouvelle_text = tfidf_vectorizer.transform(nouvelle_donnee["Payload"] + " " + nouvelle_donnee["URL"] + " " + nouvelle_donnee["Endpoint"]).toarray()
X_nouvelle = pd.concat([nouvelle_donnee[["Status_Code", "Response_Size"]], pd.DataFrame(X_nouvelle_text)], axis=1)

prediction = model.predict(X_nouvelle)
print("Type d'attaque prédit :", label_encoder.inverse_transform(prediction))
