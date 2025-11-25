"""
Multi-Agent System for Stock Market News Research and Article Writing
Uses LangChain and LangGraph to coordinate three agents:
1. Research Agent - gathers latest news from financial sources
2. Chief Editor Agent - reviews and validates information
3. Writer Agent - creates final article
"""

from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_tavily import TavilySearch
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Initialize the LLM (OpenAI)
llm = ChatOpenAI(
    model="gpt-4",  # or "gpt-4-turbo" or "gpt-3.5-turbo"
    api_key=os.getenv("OPENAI_API_KEY"),  # Set your API key
    temperature=0.7
)

# Initialize Tavily Search for real-time web search
tavily_search = TavilySearch(
    api_key=os.getenv("TAVILY_API_KEY"),
    max_results=5
)

def search_financial_news(query: str) -> str:
    """Search for current financial news using Tavily"""
    try:
        # Tavily returns a dictionary with 'results' key
        response = tavily_search.invoke({"query": query})

        # Debug: print the structure
        print(f"   Results type: {type(response)}")
        print(f"   Keys: {response.keys() if isinstance(response, dict) else 'Not a dict'}")

        if not response:
            return "No results found for this query."

        # Extract results from the response dictionary
        results = response.get('results', [])

        if not results:
            return f"No results found. Response: {response}"

        formatted_results = []
        for i, result in enumerate(results, 1):
            title = result.get('title', result.get('name', 'No title'))
            url = result.get('url', 'No URL')
            content = result.get('content', result.get('snippet', result.get('description', 'No content')))

            formatted_results.append(
                f"{i}. {title}\n"
                f"   Source: {url}\n"
                f"   Content: {content}\n"
            )

        return "\n".join(formatted_results) if formatted_results else "No formatted results available."

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"   ERROR: {error_details}")
        return f"Error searching '{query}': {str(e)}\n{error_details}"

# Define the state that will be passed between agents
class AgentState(TypedDict):
    research_results: str
    editor_feedback: str
    final_article: str
    messages: Annotated[list, operator.add]

# Agent 1: Research Agent
def research_agent(state: AgentState) -> AgentState:
    """
    Research agent that gathers latest news affecting stock market
    from sources like CNBC, Bloomberg, etc.
    """
    today = datetime.now().strftime("%B %d, %Y")

    print("🔍 Research Agent: Searching for latest financial news...\n")

    # Search for current financial news
    queries = [
        f"stock market news today {today}",
        f"breaking financial news {today} CNBC Bloomberg",
        f"federal reserve interest rates {today}",
        f"earnings reports stock market {today}"
    ]

    all_results = []
    for query in queries:
        print(f"   Searching: {query}")
        results = search_financial_news(query)
        all_results.append(results)

    combined_search_results = "\n\n".join(all_results)

    system_prompt = f"""You are a financial research analyst specializing in finding 
    breaking news that affects stock markets. Today is {today}.
    
    Your job is to:
    1. Analyze the search results provided
    2. Identify the most important financial news from TODAY
    3. Focus on news from reputable sources like CNBC, Bloomberg, Reuters, WSJ
    4. Highlight potential market impact
    5. Provide 3-5 key news items with brief summaries and sources
    
    Format your findings clearly with sources and potential market implications."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""Based on these search results from TODAY ({today}), 
        summarize the most important financial news affecting stock markets:
        
        SEARCH RESULTS:
        {combined_search_results}
        
        Provide a clear summary of the top 3-5 stories with their sources and market implications.""")
    ]

    response = llm.invoke(messages)

    return {
        **state,
        "research_results": response.content,
        "messages": [("research_agent", response.content)]
    }

# Agent 2: Chief Editor Agent
def editor_agent(state: AgentState) -> AgentState:
    """
    Chief editor reviews research findings for accuracy, relevance, and completeness
    """
    system_prompt = """You are a Chief Editor at a financial publication. Your role is to:
    1. Review research findings for accuracy and credibility
    2. Identify any gaps or missing critical information
    3. Verify that sources are reputable
    4. Suggest angles or perspectives for the article
    5. Ensure balanced coverage
    
    Provide constructive feedback and approve the direction for the article."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""Review this research and provide editorial guidance:

RESEARCH FINDINGS:
{state['research_results']}

Provide your editorial review including:
- Assessment of research quality
- Key angles to emphasize
- Any missing information
- Recommended article structure
- Approval or concerns""")
    ]

    response = llm.invoke(messages)

    return {
        **state,
        "editor_feedback": response.content,
        "messages": [("editor_agent", response.content)]
    }

# Agent 3: Writer Agent
def writer_agent(state: AgentState) -> AgentState:
    """
    Writer creates a one-page article based on research and editorial guidance
    """
    system_prompt = """You are a professional financial journalist. Your task is to:
    1. Write a compelling one-page article (approximately 500-600 words)
    2. Include a strong headline
    3. Open with the most important information
    4. Provide context and analysis
    5. Write in clear, professional language
    6. Cite sources appropriately
    7. End with market implications or forward-looking insights
    
    Make it engaging but professional, suitable for financial news publication."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""Write a one-page article based on this research and editorial guidance:

RESEARCH:
{state['research_results']}

EDITORIAL GUIDANCE:
{state['editor_feedback']}

Create a polished, publication-ready article.""")
    ]

    response = llm.invoke(messages)

    return {
        **state,
        "final_article": response.content,
        "messages": [("writer_agent", response.content)]
    }

# Build the workflow graph
def create_workflow():
    """Create the LangGraph workflow connecting all agents"""
    workflow = StateGraph(AgentState)

    # Add nodes for each agent
    workflow.add_node("research", research_agent)
    workflow.add_node("editor", editor_agent)
    workflow.add_node("writer", writer_agent)

    # Define the flow: research -> editor -> writer -> end
    workflow.set_entry_point("research")
    workflow.add_edge("research", "editor")
    workflow.add_edge("editor", "writer")
    workflow.add_edge("writer", END)

    return workflow.compile()

# Main execution
def run_news_pipeline():
    """Execute the multi-agent pipeline"""
    print("🚀 Starting Multi-Agent News Research System...\n")

    # Create and run workflow
    app = create_workflow()

    # Initialize state
    initial_state = {
        "research_results": "",
        "editor_feedback": "",
        "final_article": "",
        "messages": []
    }

    # Run the workflow
    result = app.invoke(initial_state)

    # Display results
    print("=" * 80)
    print("📊 RESEARCH AGENT OUTPUT")
    print("=" * 80)
    print(result["research_results"])
    print("\n")

    print("=" * 80)
    print("✏️ CHIEF EDITOR FEEDBACK")
    print("=" * 80)
    print(result["editor_feedback"])
    print("\n")

    print("=" * 80)
    print("📰 FINAL ARTICLE")
    print("=" * 80)
    print(result["final_article"])
    print("\n")

    return result

if __name__ == "__main__":
    # Make sure to set your API keys first:
    # export OPENAI_API_KEY='your-api-key-here'
    # export TAVILY_API_KEY='your-tavily-api-key-here'

    # Get Tavily API key free at: https://tavily.com/

    # Install the correct package:
    # uv pip install langchain-tavily

    result = run_news_pipeline()

    # Optional: Save article to file
    with open("market_news_article.txt", "w") as f:
        f.write(result["final_article"])
    print("✅ Article saved to 'market_news_article.txt'")