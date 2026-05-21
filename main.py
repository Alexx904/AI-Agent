from dotenv import load_dotenv # Carica le variabili d'ambiente dal file .env
from pydantic import BaseModel # Per definire modelli di dati con validazione
from langchain_openai import ChatOpenAI # Importa il modello di chat di OpenAI
from langchain_anthropic import ChatAnthropic # Importa il modello di chat di Anthropic
from langchain_core.prompts import ChatPromptTemplate # Per creare template di prompt per i modelli di chat
from langchain_core.outputs_parsers import PydanticOutputParser # Per analizzare le uscite dei modelli di chat con Pydantic

load_dotenv()


class ResearchResponse(BaseModel): # Definisce un modello di dati per la risposta della ricerca
    topic: str # Il tema della ricerca
    summary: str # Un riassunto dei risultati della ricerca
    sources: list[str] # Una lista di fonti utilizzate per la ricerca
    tools_used: list[str] # Una lista di strumenti utilizzati per la ricerca


llm = ChatAnthropic(model="claude-2") # Inizializza il modello di chat di Anthropic
parser = PydanticOutputParser(pydantic_object=ResearchResponse) # Crea un parser per analizzare le uscite del modello di chat di Anthropic

prompt = ChatPromptTemplate.from_messages(
    [ # Crea un template di prompt per il modello di chat di Anthropic
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use necessary tools.
            Wrap the output in this format and provde no other text\n{format_instructions}
            """,
        ), # Il messaggio di sistema definisce il ruolo del modello di chat di Anthropic e fornisce istruzioni su come formattare l'output
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions()) # Parzializza il template di prompt con le istruzioni di formattazione del parser




############################## Esempio di utilizzo dei modelli di chat
# llm2 = ChatOpenAI(model="gpt-3.5-turbo") # Inizializza il modello di chat di OpenAI
# response = llm2.invoke("What is the meaning of life?") # Esegue una richiesta al modello di chat di OpenAI
# print(response) # Stampa la risposta del modello di chat di OpenAI
