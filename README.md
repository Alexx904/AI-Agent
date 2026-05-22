# My First AI Agent

A Python-based research assistant powered by LangChain and AI models (OpenAI/Anthropic). This intelligent agent leverages multiple tools to conduct web research, gather information from Wikipedia, and save findings to files.

## 🎯 Overview

This project demonstrates how to build an AI agent that can:
- **Conduct web searches** using DuckDuckGo
- **Query Wikipedia** for structured information
- **Save research outputs** to files with timestamps
- **Parse structured responses** using Pydantic models
- **Support multiple LLM providers** (OpenAI and Anthropic)

The agent takes user queries and autonomously decides which tools to use to generate comprehensive research responses.

## 🚀 Features

### Core Capabilities
- **Web Search Tool**: Search the internet using DuckDuckGo
- **Wikipedia Tool**: Query Wikipedia for reliable information
- **File Saving Tool**: Persist research results to text files with automatic timestamps
- **Structured Output**: Results are parsed into a well-defined data model

### AI Model Support
- **Anthropic Claude**: Primary LLM for the agent
- **OpenAI GPT**: Alternative LLM option (example code included)

### Smart Agent Design
- Tool-calling agent that intelligently selects appropriate tools for queries
- Pydantic-based response validation for structured output
- Verbose execution mode for transparency and debugging

## 📋 Requirements

All dependencies are listed in `requirements.txt`:

```
langchain              # LangChain framework for agent building
wikipedia              # Wikipedia API access
langchain-community    # Community tools and utilities
langchain-openai       # OpenAI integration
langchain-anthropic    # Anthropic integration
python-dotenv          # Environment variable management
pydantic              # Data validation and parsing
ddgs                  # DuckDuckGo search wrapper
langchain-classic     # Classic LangChain agent components
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Alexx904/My-First-AI-Agent.git
   cd My-First-AI-Agent
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the root directory with your API keys:
   ```
   ANTHROPIC_API_KEY=your_anthropic_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```

## 💻 Usage

### Running the Agent

Execute the main script:
```bash
python main.py
```

The agent will prompt you to enter a research query:
```
What can I help you research? [Your query here]
```

### Example Queries
- "What is quantum computing and its applications?"
- "Tell me about the latest developments in AI"
- "Research the history and impact of machine learning"

### Output

The agent returns a structured `ResearchResponse` containing:
- **topic**: The research topic
- **summary**: A comprehensive summary of findings
- **sources**: List of sources used
- **tools_used**: Which tools the agent utilized

Results are also saved to `output.txt` with timestamps for future reference.

## 🏗️ Project Structure

```
.
├── main.py              # Main agent implementation
├── tools.py             # Custom tool definitions
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (not tracked)
└── output.txt          # Research results (generated)
```

### File Descriptions

**`main.py`**
- Initializes the Anthropic Claude LLM
- Sets up the research assistant agent
- Configures prompt template and output parsing
- Handles user input and displays results
- Includes example code for using OpenAI models

**`tools.py`**
- Implements web search via DuckDuckGo
- Configures Wikipedia query tool
- Creates custom file-saving tool
- Wraps all tools in LangChain Tool interface

## 🔧 Configuration

### LLM Selection

The agent currently uses **Anthropic Claude-2**. To switch to OpenAI:

```python
# In main.py, replace:
llm = ChatAnthropic(model="claude-2")

# With:
llm = ChatOpenAI(model="gpt-3.5-turbo")
```

### Tool Parameters

Modify Wikipedia results in `tools.py`:
```python
api_wrapper = WikipediaAPIWrapper(
    top_k_results=1,          # Number of results
    doc_content_chars_max=100 # Max characters per result
)
```

## 🔐 Security Notes

- **Never commit `.env` files** containing API keys to version control
- Use environment variables for all sensitive credentials
- The `.gitignore` should include `.env`
- Keep API keys private and rotate them regularly

## 📝 Response Format

The agent structures its output in JSON format with validation:

```json
{
  "topic": "Research Topic",
  "summary": "Detailed summary of findings...",
  "sources": ["source1.com", "source2.org"],
  "tools_used": ["search_web", "wikipedia"]
}
```

## 🐛 Troubleshooting

### Common Issues

**"API Key Error"**
- Ensure `.env` file exists and contains valid API keys
- Verify `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` is set correctly

**"DuckDuckGo Search Not Working"**
- Note: The library uses `ddgs` (not `duckduckgo_search`)
- Ensure stable internet connection
- Check DuckDuckGo service availability

**"Wikipedia Results Empty"**
- Adjust `top_k_results` or `doc_content_chars_max` in `tools.py`
- Try more specific search queries

**"Parsing Error"**
- Check that the LLM output follows the expected format
- Review the `ResearchResponse` Pydantic model
- Enable verbose mode for debugging

## 🎓 Learning Resources

This project demonstrates:
- Building multi-tool AI agents with LangChain
- Integrating multiple LLM providers
- Creating custom tools for agent workflows
- Data validation with Pydantic
- Prompt engineering and templating
- File I/O with timestamps
- Environment variable management in Python

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for improvements, bug fixes, or new features.

## 📄 License

This project is open source. Check the repository for license details.

## 👤 Author

**Alexx904** - [GitHub Profile](https://github.com/Alexx904)

## 🔗 Related Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Anthropic Claude API](https://docs.anthropic.com/)
- [OpenAI API](https://platform.openai.com/docs/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [DuckDuckGo API](https://duckduckgo.com/)

## 📞 Support

For issues, questions, or suggestions, please open an issue on the GitHub repository.

---

**Happy researching! 🔍✨**
