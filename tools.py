from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun # Per utilizzare strumenti di ricerca su Wikipedia e DuckDuckGo
from langchain_community.utilities import WikipediaAPIWrapper # Per utilizzare l'API di Wikipedia
from langchain_core.tools import Tool # Per creare strumenti personalizzati
from datetime import datetime # Per lavorare con date e orari

search = DuckDuckGoSearchRun() # Inizializza lo strumento di ricerca DuckDuckGo
search_tool = tool = Tool(
    name="search_web", # Il nome dello strumento, che può essere utilizzato nel template di prompt per chiamare lo strumento
    func=search.run, # La funzione che viene eseguita quando lo strumento viene chiamato, in questo caso la funzione di ricerca di DuckDuckGo
    description="search the web for information", # Una descrizione dello strumento, che può essere utilizzata nel template di prompt per fornire istruzioni su quando utilizzare lo strumento
)