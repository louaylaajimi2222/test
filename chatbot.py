import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from elasticsearch import Elasticsearch
import time

# -------------------- Chargement et Prétraitement des Données -------------------- #

# Charger les données
DATA_PATH = "C:\\Users\\Louay\\Downloads\\Extended_Logs_Dataset_with_Attack_Labels.csv"
df = pd.read_csv(DATA_PATH)

# Encodage des labels d'attaque
label_encoder = LabelEncoder()
df["Attack Type"] = label_encoder.fit_transform(df["Attack Type"])

# Transformation du TimeCreated en timestamp
df["TimeCreated"] = pd.to_datetime(df["TimeCreated"], errors='coerce').astype(np.int64) // 10**9

# Sélection des features numériques
X = df[["TimeCreated", "EventID"]].values
y = df["Attack Type"].values

# Séparation des données en train et test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Conversion en tenseurs PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# -------------------- Définition du Modèle de Deep Learning -------------------- #

class AttackPredictionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(AttackPredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, output_size)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

# Initialisation du modèle
input_size = X_train.shape[1]
output_size = len(label_encoder.classes_)
model = AttackPredictionModel(input_size, output_size)

# Définition de la fonction de perte et de l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)

# -------------------- Entraînement du Modèle -------------------- #

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    outputs = torch.log_softmax(outputs, dim=1)
    loss = criterion(outputs, y_train)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Sauvegarde du modèle
torch.save({'model_state_dict': model.state_dict(), 'label_encoder': label_encoder.classes_}, "attack_prediction_model.pth")

# -------------------- Connexion à Elasticsearch -------------------- #

ELK_HOST = "http://192.168.179.151:9200"
ELK_USER = "elastic"
ELK_PASSWORD = "elastic"

try:
    es = Elasticsearch([ELK_HOST], basic_auth=(ELK_USER, ELK_PASSWORD))
    if es.ping():
        print("✅ Connexion à ELK réussie !")
    else:
        print("❌ Échec de connexion à ELK !")
        exit()
except Exception as e:
    print(f"❌ Erreur de connexion à ELK : {e}")
    exit()

# -------------------- Détection et Analyse des Attaques -------------------- #

attack_mapping = {
    "4768": "possible_kerbrute_or_asreproast",
    "4771": "kerberoasting_logs",
    "4625": "ntlm_logs",
    "4769": "asreproast",
    "4782": "golden_ticket",
    "4672": "pth_logs",
    "kerberos-authentication-ticket-requested": "kerbrute_detected",
    "*kerbrute*": "kerbrute_detected",
    "krb5asrep": "asreproast",
    "0x6": "golden_ticket",
    "failure": "pth_logs"
}

previously_detected_attacks = set()

def trigger_alert(attack_name):
    print(f'⚠️ Alerte : Nouvelle attaque détectée -> {attack_name}')

# Fonction de prédiction d'attaque avec ML
def predict_attack(event_id, timestamp):
    """Prédit l'attaque en utilisant le modèle ML"""
    model.eval()
    try:
        input_data = torch.tensor([[float(timestamp), float(event_id)]], dtype=torch.float32)
        output = model(input_data)
        _, predicted = torch.max(output, 1)
        attack_label = label_encoder.inverse_transform([predicted.item()])[0]
        return attack_label
    except Exception as e:
        return f"Erreur lors de la prédiction: {e}"

def get_attack_logs(index="winlogbeat-*", size=100):
    query = {"size": size, "query": {"bool": {"must": []}}, "sort": [{"@timestamp": "desc"}]}
    try:
        response = es.search(index=index, body=query)
        logs = [hit["_source"] for hit in response["hits"]["hits"]]
        attack_results = []

        for log in logs:
            event_code = str(log.get("event", {}).get("code", ""))
            timestamp = log.get("TimeCreated", 0)
            message = log.get("message", "").lower()
            attack_name = attack_mapping.get(event_code, None)

            # Vérification pour différencier AS-REP Roasting et Kerbrute
            if event_code == "4769":
                if "krb5asrep" in message or "asrep" in message:
                    attack_name = "asreproast"
                else:
                    attack_name = "kerbrute_userenum"

            # Si aucune attaque n'est identifiée, utiliser le modèle ML
            if attack_name is None:
                attack_name = predict_attack(event_code, timestamp)

            attack_results.append(attack_name)

        new_attacks = []
        for attack in attack_results:
            if attack not in previously_detected_attacks:
                trigger_alert(attack)
                new_attacks.append(attack)
                previously_detected_attacks.add(attack)
        
        return new_attacks if new_attacks else ["✅ Aucun nouveau signe d'attaque trouvé."]
    except Exception as e:
        return [f"❌ Erreur lors de la récupération des logs d'attaques : {e}"]

# -------------------- Chatbot Cybersécurité -------------------- #

def handle_conversation():
    print("🔍 Chatbot Cybersécurité - Tape 'exit' pour quitter.")
    while True:
        user_input = input("You: ").lower()
        if user_input == "exit":
            print("Goodbye! 🔐")
            break
        if "attaque" in user_input or "asrep" in user_input:
            logs = get_attack_logs()
            print("🛡️ Bot:", logs)
            continue

# -------------------- Exécution -------------------- #
if __name__ == "__main__":
    handle_conversation()
