from langchain.agents import initialize_agent, AgentType
from langchain_openai import OpenAI
from langchain.tools import Tool
from langchain_community.utilities import GoogleSearchAPIWrapper
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Get keys from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLEAPI_KEY = os.getenv("GOOGLEAPI_KEY")
GOOGLECSE_ID = os.getenv("GOOGLECSE_ID")

# Initialize LLM
llm = OpenAI(model="llama3-8b-8192", api_key=GROQ_API_KEY)

# Set up search tool
search = GoogleSearchAPIWrapper(
    google_api_key=GOOGLEAPI_KEY,
    google_cse_id=GOOGLECSE_ID
)

search_tool = Tool(
    name="Google Search",
    func=search.results,
    description="Search google for academic papers and latest research."
)

# Define a summarizer function
def summarize_with_llm(text):
    prompt = f"Summarize the following research finding:\n\n{text}\n\nProvide a short and clear summary."
    response = llm.invoke(prompt)
    return response

# Calculator tool
calc_tool = Tool(
    name="Calculator",
    func=lambda x: eval(x),
    description="Performs calculations"
)

# Initialize agent
agent = initialize_agent(
    tools=[search_tool, calc_tool],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

# Run a search query
query = "Find recent research on wireless networks."
search_results = search.results(query, num_results=5)
print(search_results)