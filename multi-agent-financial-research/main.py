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
import os

from dotenv import load_dotenv

load_dotenv()

# Initialize the LLM (OpenAI)
llm = ChatOpenAI(
    model="gpt-4",  # or "gpt-4-turbo" or "gpt-3.5-turbo"
    api_key=os.getenv("OPENAI_API_KEY"),  # Set your API key
    temperature=0.7
)


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
    system_prompt = """You are a financial research analyst specializing in finding 
    breaking news that affects stock markets. Your job is to:
    1. Identify the most important financial news from the last 24 hours
    2. Focus on news from reputable sources like CNBC, Bloomberg, Reuters, WSJ
    3. Highlight potential market impact
    4. Provide 3-5 key news items with brief summaries

    Format your findings clearly with sources and potential market implications."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content="Research the latest financial news that could impact stock markets today. Focus on major economic indicators, corporate earnings, Fed policy, geopolitical events, and sector-specific news.")
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
    # Make sure to set your API key first:
    # export OPENAI_API_KEY='your-api-key-here'

    # Note: For real web searching, you would integrate with tools like:
    # - Tavily Search API
    # - SerpAPI
    # - Custom web scraping
    # This example uses GPT's knowledge, but you can add tool calling

    result = run_news_pipeline()

    # Optional: Save article to file
    with open("market_news_article.txt", "w") as f:
        f.write(result["final_article"])
    print("✅ Article saved to 'market_news_article.txt'")