from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
import os
import sys
from datetime import datetime

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Verifica se la chiave API è stata trovata
if not api_key:
    print("Errore: La chiave API di OpenAI non è stata trovata. Assicurati di averla inserita nel file .env.")
    sys.exit(1)

try:
    user_temp = float(input("Inserisci la temperatura (0.0 - 1.0): "))
    if not (0.0 <= user_temp <= 1.0):
        print("Valore non valido. Inserisci un valore tra 0.0 e 1.0.")
        sys.exit(1)
except ValueError:
    print("Valore non valido.")
    sys.exit(1)

# Istanzia il modello OpenAI
chat = ChatOpenAI(api_key=api_key, temperature=user_temp)

# Definire la lingua del bot
user_lang = input("In quale lingua vuoi chattare? (es. 'it' per italiano, 'en' per inglese): ").strip().lower()
system_prompt = SystemMessage(content=f"Sei un assistente virtuale che parla {user_lang}. Rispondi alle domande dell'utente in {user_lang}.")

mode = input("Vuoi che il bot risponda in MAIUSCOLO o minuscolo? ").strip().lower()
if mode not in ["maiuscolo", "minuscolo"]:
    print("Valore non valido. Rispondi con 'MAIUSCOLO' o 'minuscolo'.")
    mode = "normale"

# Loop di chat
def chat_loop():
    print("Inizia la chat (digita 'esci/exit' per uscire):")

    greeting = "Ciao! Sono un assistente virtuale. Come posso aiutarti oggi?"

    if mode == "maiuscolo":
        greeting = greeting.upper()
    elif mode == "minuscolo":
        greeting = greeting.lower()

    print(f"AI [{datetime.now().strftime('%H:%M')}]: {greeting}")

    chat_history = [system_prompt]

    while True:
        user_input = input("Tu: ")
        if user_input.lower() == "esci" or user_input.lower() == "exit":
            print("Chat terminata.")
            break
        
        chat_history.append(HumanMessage(content=user_input)) # Aggiunge il messaggio dell'utente alla cronologia della chat

        response = chat.invoke(chat_history) # Ottiene la risposta del modello OpenAI in base alla cronologia della chat

        chat_history.append(response) # Aggiunge la risposta del modello alla cronologia della chat

        output_text = response.content # Limita la risposta a 200 caratteri per evitare output troppo lunghi      
        if mode == "maiuscolo":
            output_text = output_text.upper()
        elif mode == "minuscolo":
            output_text = output_text.lower()

        print(f"AI [{datetime.now().strftime('%H:%M')}]: {output_text[:200]}")

# Avvia il loop di chat
chat_loop()