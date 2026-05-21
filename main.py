from dotenv import load_dotenv # Carica le variabili d'ambiente dal file .env
from pydantic import BaseModel # Per definire modelli di dati con validazione
from langchain_openai import ChatOpenAI # Importa il modello di chat di OpenAI
from langchain_anthropic import ChatAnthropic # Importa il modello di chat di Anthropic

load_dotenv()

llm2 = ChatOpenAI(model="gpt-3.5-turbo") # Inizializza il modello di chat di OpenAI
llm = ChatAnthropic(model="claude-2") # Inizializza il modello di chat di Anthropic

response = llm.invoke("What is the meaning of life?") # Esegue una richiesta al modello di chat di Anthropic
print(response) # Stampa la risposta ottenuta