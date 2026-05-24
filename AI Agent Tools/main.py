from dotenv import load_dotenv # Carica le variabili d'ambiente dal file .env
from pydantic import BaseModel # Per definire modelli di dati con validazione
from langchain_openai import ChatOpenAI # Importa il modello di chat di OpenAI
from langchain_anthropic import ChatAnthropic # Importa il modello di chat di Anthropic
from langchain_core.prompts import ChatPromptTemplate # Per creare template di prompt per i modelli di chat
from langchain_core.output_parsers import PydanticOutputParser # Per analizzare le uscite dei modelli di chat con Pydantic
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor # Per creare agenti che possono utilizzare strumenti durante la conversazione
from tools import search_tool, wiki_tool, save_tool # Importa gli strumenti definiti in tools.py creati per eseguire ricerche su DuckDuckGo e Wikipedia e per salvare i dati in un file di testo    

load_dotenv()

class ResearchResponse(BaseModel): # Definisce un modello di dati per la risposta della ricerca
    topic: str # Il tema della ricerca
    summary: str # Un riassunto dei risultati della ricerca
    sources: list[str] # Una lista di fonti utilizzate per la ricerca
    tools_used: list[str] # Una lista di strumenti utilizzati per la ricerca


llm = ChatAnthropic(model="claude-2", temperature=1.0) # Inizializza il modello di chat di Anthropic
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
        ("placeholder", "{chat_history}"), # Il messaggio di placeholder è utilizzato per inserire la cronologia della chat, che può essere utilizzata per tenere traccia delle conversazioni precedenti
        ("human", "{query} {name}"), # Il messaggio umano è utilizzato per inserire la query dell'utente, che è la domanda o richiesta a cui il modello di chat di Anthropic deve rispondere
        ("placeholder", "{agent_scratchpad}"), # Il messaggio di placeholder è utilizzato per inserire la cronologia della chat e il "scratchpad" dell'agente, che può essere utilizzato per tenere traccia delle azioni dell'agente durante la conversazione
    ]
).partial(format_instructions=parser.get_format_instructions()) # Parzializza il template di prompt con le istruzioni di formattazione del parser

tools = [search_tool, wiki_tool, save_tool] # Crea una lista di strumenti che l'agente può utilizzare durante la conversazione, in questo caso solo lo strumento di ricerca su DuckDuckGo e Wikipedia
agent = create_tool_calling_agent(
    llm=llm, # Il modello di chat di Anthropic che l'agente utilizzerà per rispondere alle query dell'utente
    prompt=prompt, # Il template di prompt che l'agente utilizzerà per formattare le risposte
    tools=tools, # Qui puoi aggiungere strumenti che l'agente può utilizzare durante la conversazione
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) # Crea un esecutore per l'agente, che gestirà l'esecuzione delle azioni dell'agente durante la conversazione, verbose=True permette di vedere i dettagli dell'esecuzione
query = input("What can I help you research? ") # Chiede all'utente di inserire una query
raw_response = agent_executor.invoke({"query": query, "name": "Alice"}) # Esegue una richiesta all'agente, passando la query dell'utente e un nome (che può essere utilizzato nel template di prompt)
# print(raw_response) # Stampa la risposta grezza dell'agente

try:
    structured_response = parser.parse(raw_response.get("output")[0]["text"]) # Analizza la risposta grezza dell'agente utilizzando il parser Pydantic per ottenere una risposta strutturata
    # print(structured_response.topic) # Stampa il topic della ricerca dalla risposta strutturata, se tolgo .topic ottengo tutta la risposta strutturata
    print(structured_response) # Stampa la risposta strutturata completa
except Exception as e:
    print(f"Error parsing response", e, "Raw response - ", raw_response) # Stampa un messaggio di errore se c'è un problema durante l'analisi della risposta, insieme alla risposta grezza per il debug

## Esempio di utilizzo dei modelli di chat
# llm2 = ChatOpenAI(model="gpt-3.5-turbo") # Inizializza il modello di chat di OpenAI
# response = llm2.invoke("What is the meaning of life?") # Esegue una richiesta al modello di chat di OpenAI
# print(response) # Stampa la risposta del modello di chat di OpenAI
#
#           
#
#
#
#
#
#
#
