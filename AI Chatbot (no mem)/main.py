from langchain_openai import ChatOpenAI # Serve per interagire con il modello OpenAI, in questo caso per creare un'istanza del modello e inviare messaggi
from langchain_core.messages import HumanMessage # Serve per creare messaggi umani da inviare al modello
from dotenv import load_dotenv # Serve per caricare le variabili d'ambiente da un file .env
import os # Serve per accedere alle variabili d'ambiente, in questo caso la chiave API di OpenAI
import sys # Serve per uscire dal programma in caso di errore nella chiave API

# Carica la chiave API da file .env
load_dotenv() # Serve per caricare le variabili d'ambiente dal file .env
api_key = os.getenv("OPENAI_API_KEY") # Recupera la chiave API dall'ambiente


if not api_key:
    print("Errore: La chiave API di OpenAI non è stata trovata. Assicurati di averla inserita nel file .env.")
    sys.exit(1)

try:
    user_temp = float(input("Inserisci la temperatura (0.0 - 1.0): ")) # Chiede all'utente di inserire una temperatura per il modello, con un valore predefinito di 1.0
except ValueError:
    print("Valore non valido.")
    sys.exit(1)

# Istanzia il modello OpenAI
chat = ChatOpenAI(api_key=api_key, temperature=user_temp) # Crea un'istanza del modello OpenAI con la chiave API e la temperatura inserita dall'utente

# Loop di chat
def chat_loop():
    print("Inizia la chat (digita 'esci/exit' per uscire):")

    # Saluto iniziale
    greeting = "Ciao! Sono un assistente virtuale. Come posso aiutarti oggi?"
    print(f"AI: {greeting}")

    while True:
        user_input = input("Tu: ")
        if user_input.lower() == "esci" or user_input.lower() == "exit":
            print("Chat terminata.")
            break
        
        # Crea un messaggio umano e ottieni la risposta del modello
        msg = HumanMessage(content=user_input) # Crea un messaggio umano con il contenuto dell'input dell'utente
        response = chat.invoke([msg]) # Invia il messaggio ad chat = ChatOpenAI e ottieni la risposta che viene stampata a video
        
        print(f"AI: {response.content}")

# Avvia il loop di chat
chat_loop()