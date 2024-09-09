import random
import json
import streamlit as st
#import torch
import time

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Charger le modèle et les intentions
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r', encoding='utf-8') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Initialiser l'application Streamlit
st.title("Blessing Chatbot")
bot_name = "Blessing"

# Initialiser l'historique des messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Afficher l'historique des messages
for message in st.session_state.messages:
    if message['role'] == 'user':
        st.chat_message(message['role']).markdown(f"**Vous:** {message['content']}")
    else:
        st.chat_message(message['role']).markdown(f"**{bot_name}:** {message['content']}")

# Saisie de l'utilisateur
user_input = st.chat_input("Pose moi une question")

if user_input:
    # Ajouter l'entrée de l'utilisateur à l'historique des messages
    st.session_state.messages.append({'role': 'user', 'content': user_input})

    # Afficher un message de "chargement" pendant le délai de recherche
    with st.spinner('Je cherche...'):
        time.sleep(2)  # Simule un délai de recherche de 2 secondes

        # Tokeniser et traiter l'entrée
        sentence = tokenize(user_input)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        # Obtenir la prédiction du modèle
        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        # Déterminer la réponse
        if prob.item() > 0.75:
            response = "Je ne comprends pas..."
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    response = random.choice(intent['responses'])
                    break
        else:
            response = "Je ne comprends pas..."

    # Ajouter la réponse du bot à l'historique des messages
    st.session_state.messages.append({'role': 'assistant', 'content': response})

    # Reafficher les messages pour mettre à jour l'affichage
    for message in st.session_state.messages:
        if message['role'] == 'user':
            st.chat_message(message['role']).markdown(f"**Vous:** {message['content']}")
        else:
            st.chat_message(message['role']).markdown(f"**{bot_name}:** {message['content']}")

    # S'assurer que le champ de saisie est réinitialisé
    st.experimental_rerun()

from sklearn.metrics import accuracy_score, precision_score
import numpy as np

# Exemple de réponses réelles et prédites
y_true = ["réponse1", "réponse2", "réponse1", "réponse3"]
y_pred = ["réponse1", "réponse2", "réponse3", "réponse3"]

# Calcul de l'accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Calcul de la précision pour chaque classe
precision = precision_score(y_true, y_pred, average='macro')  # 'macro' pour la précision moyenne
print(f"Precision: {precision:.2f}")


