from langchain_openai import ChatOpenAI # Serve per interagire con il modello OpenAI, in questo caso per creare un'istanza del modello e inviare messaggi
from langchain_core.messages import HumanMessage, SystemMessage # Serve per creare messaggi umani e di sistema, che vengono utilizzati per comunicare con il modello OpenAI e impostare il contesto della conversazione
from dotenv import load_dotenv # Serve per caricare le variabili d'ambiente da un file .env
import os # Serve per accedere alle variabili d'ambiente, in questo caso la chiave API di OpenAI
import sys # Serve per uscire dal programma in caso di errore nella chiave API
from datetime import datetime # Serve per ottenere la data e l'ora corrente

# Carica la chiave API da file .env
load_dotenv() # Serve per caricare le variabili d'ambiente dal file .env
api_key = os.getenv("OPENAI_API_KEY") # Recupera la chiave API dall'ambiente


# Verifica se la chiave API è stata trovata
if not api_key:
    print("Errore: La chiave API di OpenAI non è stata trovata. Assicurati di averla inserita nel file .env.")
    sys.exit(1)

try:
    user_temp = float(input("Inserisci la temperatura (0.0 - 1.0): ")) # Chiede all'utente di inserire una temperatura per il modello
    if not (0.0 <= user_temp <= 1.0):
        print("Valore non valido. Inserisci un valore tra 0.0 e 1.0.")
        sys.exit(1)
except ValueError:
    print("Valore non valido.")
    sys.exit(1)

# Istanzia il modello OpenAI
chat = ChatOpenAI(api_key=api_key, temperature=user_temp) # Crea un'istanza del modello OpenAI con la chiave API e la temperatura inserita dall'utente

# Definire la lingua del bot
user_lang = input("In quale lingua vuoi chattare? (es. 'it' per italiano, 'en' per inglese): ").strip().lower() # Chiede all'utente di scegliere la lingua per la chat
system_prompt = SystemMessage(content=f"Sei un assistente virtuale che parla {user_lang}. Rispondi alle domande dell'utente in {user_lang}.") # Crea un messaggio di sistema per impostare la lingua del bot

# Loop di chat
def chat_loop():
    print("Inizia la chat (digita 'esci/exit' per uscire):")

    # Saluto iniziale
    greeting = "Ciao! Sono un assistente virtuale. Come posso aiutarti oggi?"
    print(f"AI [{datetime.now().strftime('%H:%M')}]: {greeting}") # Stampa un messaggio di benvenuto con l'ora corrente

    while True:
        user_input = input("Tu: ")
        if user_input.lower() == "esci" or user_input.lower() == "exit":
            print("Chat terminata.")
            break
        
        # Crea un messaggio umano e ottieni la risposta del modello
        msg = [system_prompt, HumanMessage(content=user_input)] # Crea un messaggio umano con il contenuto dell'input dell'utente
        response = chat.invoke(msg) # Invia il messaggio ad chat = ChatOpenAI e ottieni la risposta che viene stampata a video
        
        print(f"AI [{datetime.now().strftime('%H:%M')}]: {response.content[:200]}") # Stampa la risposta del modello con l'ora corrente, limitando a 50 caratteri per evitare output troppo lunghi

# Avvia il loop di chat
chat_loop()