from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun # Per utilizzare strumenti di ricerca su Wikipedia e DuckDuckGo
from langchain_community.utilities import WikipediaAPIWrapper # Per utilizzare l'API di Wikipedia
from langchain_core.tools import Tool # Per creare strumenti personalizzati
from datetime import datetime # Per lavorare con date e orari

#################################################################### Creazione di uno strumento personalizzato per salvare i dati in un file di testo

def save_to_txt(data: str, filename: str = "output.txt"): # Definisce una funzione per salvare i dati in un file di testo, con un timestamp per tenere traccia di quando sono stati salvati
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") # Ottiene il timestamp corrente
    formatted_data = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n" # Formatta i dati con il timestamp

    with open(filename, "a", encoding="utf-8") as f: # Apre il file in modalità append per aggiungere nuovi dati senza sovrascrivere quelli esistenti
        f.write(formatted_data + "\n\n") # Scrive i dati formattati nel file, separandoli con nuove righe
    return f"Data saved to {filename} at {timestamp}" # Restituisce un messaggio di conferma con il nome del file e il timestamp

save_tool = Tool( # Crea uno strumento personalizzato per salvare i dati in un file di testo
    name="save_text_to_file", # Il nome dello strumento, che può essere utilizzato nel template di prompt per chiamare lo strumento
    func=save_to_txt, # La funzione che viene eseguita quando lo strumento viene chiamato, in questo caso la funzione per salvare i dati in un file di testo
    description="Saves structured text to a file", # Una descrizione dello strumento, che può essere utilizzata nel template di prompt per fornire istruzioni su quando utilizzare lo strumento
)

####################################################################

search = DuckDuckGoSearchRun() # Inizializza lo strumento di ricerca DuckDuckGo

search_tool = tool = Tool( # Crea uno strumento personalizzato per eseguire ricerche su DuckDuckGo
    name="search_web", # Il nome dello strumento, che può essere utilizzato nel template di prompt per chiamare lo strumento
    func=search.run, # La funzione che viene eseguita quando lo strumento viene chiamato, in questo caso la funzione di ricerca di DuckDuckGo
    description="search the web for information", # Una descrizione dello strumento, che può essere utilizzata nel template di prompt per fornire istruzioni su quando utilizzare lo strumento
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100) # Inizializza l'API wrapper di Wikipedia
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper) # Inizializza lo strumento di ricerca su Wikipedia utilizzando l'API wrapper

####################################################################