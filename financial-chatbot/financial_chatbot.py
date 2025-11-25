"""
Financial Investment Chatbot using LangGraph and ChatOpenAI
Features: Memory, Router, Multiple Tools, Real-time Market Data
Up-to-date as of November 24, 2025

Setup with uv:
1. Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh
2. Create project: uv init financial-chatbot && cd financial-chatbot
3. Add dependencies:
   uv add langgraph langchain-openai langchain-community duckduckgo-search python-dotenv
4. Create .env file with: OPENAI_API_KEY=your-key-here
5. Run: uv run python main.py

Or use the quick start script below.
"""

from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize ChatOpenAI
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Initialize search tool for real-time data
search_wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)
search_tool = DuckDuckGoSearchResults(api_wrapper=search_wrapper)

CURRENT_DATE = "November 24, 2025"


# Define the state
class AgentState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], add_messages]
    current_tool: str
    memory_summary: str
    search_results: str
    needs_current_data: bool


# Search Node for Real-time Data
def search_node(state: AgentState) -> AgentState:
    """Search for current market data and news"""
    query = state["messages"][-1].content

    # Extract key search terms
    search_query = f"financial market {query} November 2025"

    print(f"\n🔍 Searching for current data: {search_query[:50]}...")

    try:
        results = search_tool.run(search_query)
        state["search_results"] = results
        print(f"✅ Found current market information")
    except Exception as e:
        state["search_results"] = f"Search unavailable: {str(e)}"
        print(f"⚠️ Search failed: {str(e)}")

    return state


# Router Node
def router_node(state: AgentState) -> AgentState:
    """Routes the query to the appropriate tool and determines if search is needed"""
    last_message = state["messages"][-1].content.lower()

    # Check if current data is needed
    current_data_keywords = ["current", "today", "now", "latest", "recent", "2025",
                             "this year", "price", "rate", "trending"]
    state["needs_current_data"] = any(word in last_message for word in current_data_keywords)

    # Routing logic
    if any(word in last_message for word in ["portfolio", "allocation", "diversif", "risk level", "balance"]):
        tool = "portfolio_analysis"
    elif any(word in last_message for word in
             ["market", "trend", "economy", "sector", "inflation", "rates", "stock price"]):
        tool = "market_trends"
    else:
        tool = "investment_advice"

    state["current_tool"] = tool
    print(f"\n🔀 Router: Selected tool -> {tool} | Needs current data: {state['needs_current_data']}")
    return state


# Tool Functions
def investment_advice_tool(query: str, search_results: str = "", memory: str = "") -> str:
    """Provides general investment advice"""
    context = f"\nCurrent Date: {CURRENT_DATE}"

    if search_results:
        context += f"\n\nCurrent Market Information:\n{search_results}"

    if memory:
        context += f"\n\nConversation Context: {memory}"

    prompt = f"""You are a financial advisor providing advice as of {CURRENT_DATE}.

{context}

User Query: {query}

Provide specific, actionable investment advice considering:
- Current market conditions (if available)
- Risk management strategies
- Diversification recommendations
- Time horizon considerations

Keep response clear, practical, and current."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return f"💡 Investment Advice (as of {CURRENT_DATE}):\n\n{response.content}"


def portfolio_analysis_tool(query: str, search_results: str = "", memory: str = "") -> str:
    """Analyzes portfolio allocation and risk"""
    context = f"\nCurrent Date: {CURRENT_DATE}"

    if search_results:
        context += f"\n\nCurrent Market Data:\n{search_results}"

    if memory:
        context += f"\n\nConversation Context: {memory}"

    prompt = f"""You are a portfolio analyst providing recommendations as of {CURRENT_DATE}.

{context}

User Query: {query}

Provide detailed portfolio analysis including:
- Asset allocation percentages (stocks, bonds, alternatives, cash)
- Risk level assessment (conservative/moderate/aggressive)
- Expected returns based on current market conditions
- Rebalancing recommendations
- Sector diversification

Consider current economic environment in your recommendations."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return f"📊 Portfolio Analysis (as of {CURRENT_DATE}):\n\n{response.content}"


def market_trends_tool(query: str, search_results: str = "", memory: str = "") -> str:
    """Provides current market trend analysis"""
    context = f"\nCurrent Date: {CURRENT_DATE}"

    if search_results:
        context += f"\n\nLatest Market Information:\n{search_results}"

    if memory:
        context += f"\n\nConversation Context: {memory}"

    prompt = f"""You are a market analyst providing analysis as of {CURRENT_DATE}.

{context}

User Query: {query}

Analyze and discuss:
- Current market conditions and trends
- Sector performance (tech, healthcare, energy, financials, etc.)
- Economic indicators (inflation, interest rates, GDP)
- Investment opportunities and risks
- Short-term and long-term outlook

Use the latest market information provided to give current insights."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return f"📈 Market Trends Analysis (as of {CURRENT_DATE}):\n\n{response.content}"


# Tool Execution Nodes
def investment_advice_node(state: AgentState) -> AgentState:
    """Execute investment advice tool"""
    query = state["messages"][-1].content
    response = investment_advice_tool(
        query,
        state.get("search_results", ""),
        state.get("memory_summary", "")
    )
    state["messages"].append(AIMessage(content=response))
    return state


def portfolio_analysis_node(state: AgentState) -> AgentState:
    """Execute portfolio analysis tool"""
    query = state["messages"][-1].content
    response = portfolio_analysis_tool(
        query,
        state.get("search_results", ""),
        state.get("memory_summary", "")
    )
    state["messages"].append(AIMessage(content=response))
    return state


def market_trends_node(state: AgentState) -> AgentState:
    """Execute market trends tool"""
    query = state["messages"][-1].content
    response = market_trends_tool(
        query,
        state.get("search_results", ""),
        state.get("memory_summary", "")
    )
    state["messages"].append(AIMessage(content=response))
    return state


# Memory Node
def memory_node(state: AgentState) -> AgentState:
    """Summarizes conversation for context"""
    if len(state["messages"]) > 6:
        messages_text = "\n".join([f"{m.type}: {m.content[:100]}..." for m in state["messages"][-6:]])
        summary_prompt = f"""Summarize the key points from this financial conversation in 2-3 sentences:

{messages_text}

Focus on: investment preferences, risk tolerance, topics discussed, user goals."""

        summary = llm.invoke([HumanMessage(content=summary_prompt)])
        state["memory_summary"] = summary.content
        print(f"\n🧠 Memory Updated: {state['memory_summary'][:100]}...")

    return state


# Conditional edge functions
def should_search(state: AgentState) -> str:
    """Determine if search is needed"""
    return "search" if state.get("needs_current_data", False) else "skip_search"


def route_to_tool(state: AgentState) -> str:
    """Determine which tool node to execute"""
    return state["current_tool"]


# Build the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("router", router_node)
workflow.add_node("search", search_node)
workflow.add_node("investment_advice", investment_advice_node)
workflow.add_node("portfolio_analysis", portfolio_analysis_node)
workflow.add_node("market_trends", market_trends_node)
workflow.add_node("memory", memory_node)

# Set entry point
workflow.set_entry_point("router")

# Add conditional edge for search
workflow.add_conditional_edges(
    "router",
    should_search,
    {
        "search": "search",
        "skip_search": "investment_advice"
    }
)

# Add conditional edges from search to tools
workflow.add_conditional_edges(
    "search",
    route_to_tool,
    {
        "investment_advice": "investment_advice",
        "portfolio_analysis": "portfolio_analysis",
        "market_trends": "market_trends"
    }
)

# For skip_search path
workflow.add_conditional_edges(
    "investment_advice",
    lambda state: "memory" if state["current_tool"] == "investment_advice" else route_to_tool(state),
    {
        "memory": "memory",
        "portfolio_analysis": "portfolio_analysis",
        "market_trends": "market_trends"
    }
)

# Add edges from other tools to memory
workflow.add_edge("portfolio_analysis", "memory")
workflow.add_edge("market_trends", "memory")

# Add edge from memory to end
workflow.add_edge("memory", END)

# Compile the graph
app = workflow.compile()


# Main chat function
def chat(user_input: str, state: AgentState = None) -> tuple[str, AgentState]:
    """Process user input and return response"""
    if state is None:
        state = {
            "messages": [
                SystemMessage(content=f"You are a helpful financial investment advisor. Today is {CURRENT_DATE}.")
            ],
            "current_tool": "",
            "memory_summary": "",
            "search_results": "",
            "needs_current_data": False
        }

    # Add user message
    state["messages"].append(HumanMessage(content=user_input))

    # Run the graph
    result = app.invoke(state)

    # Get the last AI message
    response = result["messages"][-1].content

    return response, result


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print(f"Financial Investment Chatbot - Current as of {CURRENT_DATE}")
    print("=" * 70)
    print("\n🤖 Features:")
    print("  💡 Investment Advice - Personalized investment guidance")
    print("  📊 Portfolio Analysis - Asset allocation and risk assessment")
    print("  📈 Market Trends - Real-time market analysis")
    print("  🔍 Live Search - Current market data and news")
    print("  🧠 Memory - Contextual conversation tracking")
    print("\nType 'quit' to exit\n")

    state = None

    while True:
        user_input = input("\n👤 You: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Thank you for using the Financial Chatbot!")
            break

        if not user_input:
            continue

        try:
            response, state = chat(user_input, state)
            print(f"\n🤖 Assistant:\n{response}")

            # Show memory summary if available
            if state.get("memory_summary"):
                print(f"\n💭 Conversation Context: {state['memory_summary']}")

        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback

            traceback.print_exc()


# Example programmatic usage
def example_conversation():
    """Example of programmatic conversation with current data"""
    print("\n" + "=" * 70)
    print("Example Conversation with Real-time Data")
    print("=" * 70)

    questions = [
        "What are the current stock market trends?",
        "How should I allocate my portfolio with moderate risk in the current market?",
        "What's the latest on tech stocks?",
        "Should I invest in bonds given current interest rates?"
    ]

    state = None
    for question in questions:
        print(f"\n👤 Question: {question}")
        response, state = chat(question, state)
        print(f"\n🤖 Response: {response[:400]}...")
        print("-" * 70)


# Uncomment to run example
example_conversation()
