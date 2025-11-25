"""
Financial Investment Chatbot with Streamlit UI
Beautiful, modern interface for the LangGraph chatbot

Setup with uv:
uv add streamlit
uv run streamlit run app.py
"""

import streamlit as st
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

# Page configuration
st.set_page_config(
    page_title="AI Financial Advisor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
:root{
    --main-bg-color: #000000;
    --main-text-color: #ffffff;
    --muted-text-color: rgba(255,255,255,0.8);
    --card-bg: rgba(255,255,255,0.03);
    --panel-bg: rgba(255,255,255,0.02);
}

/* App background and global text color */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--main-bg-color) !important;
    color: var(--main-text-color) !important;
}

/* Ensure all common elements inherit white text and transparent backgrounds */
[data-testid="stAppViewContainer"] *, [data-testid="stSidebar"] *,
.stMarkdown, .stText, .markdown-text-container, .css-1d391kg {
    color: var(--main-text-color) !important;
    background-color: transparent !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: var(--panel-bg) !important;
    color: var(--main-text-color) !important;
}

/* Header as plain white text for visibility */
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    color: var(--main-text-color) !important;
    margin-bottom: 2rem;
    background: none;
    -webkit-text-fill-color: var(--main-text-color) !important;
}

/* Agent cards: subtle dark panels with colored left border */
.agent-card {
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1rem 0;
    border-left: 5px solid;
    color: var(--main-text-color) !important;
    background-color: var(--card-bg) !important;
}
.research-card { border-left-color: #2196F3; }
.editor-card   { border-left-color: #FF9800; }
.writer-card   { border-left-color: #4CAF50; }

/* Buttons: keep gradient and white text */
.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white !important;
    font-weight: bold;
    padding: 0.75rem;
    border-radius: 10px;
    border: none;
    font-size: 1.1rem;
}
.stButton>button:hover {
    transform: scale(1.02);
    box-shadow: 0 5px 15px rgba(0,0,0,0.5);
}

/* Alerts / info boxes: dark panel with white text */
.stAlert, .stInfo, .stSuccess, .stError {
    color: var(--main-text-color) !important;
    background-color: rgba(255,255,255,0.03) !important;
    border-color: rgba(255,255,255,0.06) !important;
}

/* Download button tweak for dark background */
.stDownloadButton>button {
    background: linear-gradient(90deg,#111827 0%,#1f2937 100%);
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_state" not in st.session_state:
    st.session_state.chat_state = None
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "tools_used" not in st.session_state:
    st.session_state.tools_used = {"investment_advice": 0, "portfolio_analysis": 0, "market_trends": 0}

# Constants
CURRENT_DATE = "November 24, 2025"

# Initialize LLM and search (cached)
@st.cache_resource
def initialize_llm():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("❌ OPENAI_API_KEY not found in .env file!")
        st.stop()
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=api_key
    )

@st.cache_resource
def initialize_search():
    search_wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)
    return DuckDuckGoSearchResults(api_wrapper=search_wrapper)

try:
    llm = initialize_llm()
    search_tool = initialize_search()
except Exception as e:
    st.error(f"❌ Error initializing services: {str(e)}")
    st.stop()

# State definition
class AgentState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], add_messages]
    current_tool: str
    memory_summary: str
    search_results: str
    needs_current_data: bool

# Node functions
def search_node(state: AgentState) -> AgentState:
    """Search for current market data"""
    query = state["messages"][-1].content
    search_query = f"financial market {query} November 2025"

    try:
        results = search_tool.run(search_query)
        state["search_results"] = results
    except Exception as e:
        state["search_results"] = ""

    return state

def router_node(state: AgentState) -> AgentState:
    """Routes the query to appropriate tool"""
    last_message = state["messages"][-1].content.lower()

    current_data_keywords = ["current", "today", "now", "latest", "recent", "2025",
                            "this year", "price", "rate", "trending"]
    state["needs_current_data"] = any(word in last_message for word in current_data_keywords)

    if any(word in last_message for word in ["portfolio", "allocation", "diversif", "risk level", "balance"]):
        tool = "portfolio_analysis"
    elif any(word in last_message for word in ["market", "trend", "economy", "sector", "inflation", "rates", "stock price"]):
        tool = "market_trends"
    else:
        tool = "investment_advice"

    state["current_tool"] = tool
    st.session_state.tools_used[tool] += 1

    return state

def investment_advice_tool(query: str, search_results: str = "", memory: str = "") -> str:
    """Provides investment advice"""
    context = f"\nCurrent Date: {CURRENT_DATE}"
    if search_results:
        context += f"\n\nCurrent Market Information:\n{search_results}"
    if memory:
        context += f"\n\nConversation Context: {memory}"

    prompt = f"""You are a financial advisor providing advice as of {CURRENT_DATE}.

{context}

User Query: {query}

Provide specific, actionable investment advice considering current market conditions, 
risk management, diversification, and time horizon. Keep response clear and practical."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return f"💡 **Investment Advice** (as of {CURRENT_DATE})\n\n{response.content}"

def portfolio_analysis_tool(query: str, search_results: str = "", memory: str = "") -> str:
    """Analyzes portfolio allocation"""
    context = f"\nCurrent Date: {CURRENT_DATE}"
    if search_results:
        context += f"\n\nCurrent Market Data:\n{search_results}"
    if memory:
        context += f"\n\nConversation Context: {memory}"

    prompt = f"""You are a portfolio analyst as of {CURRENT_DATE}.

{context}

User Query: {query}

Provide detailed portfolio analysis including asset allocation percentages, 
risk assessment, expected returns, rebalancing recommendations, and sector diversification."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return f"📊 **Portfolio Analysis** (as of {CURRENT_DATE})\n\n{response.content}"

def market_trends_tool(query: str, search_results: str = "", memory: str = "") -> str:
    """Provides market trend analysis"""
    context = f"\nCurrent Date: {CURRENT_DATE}"
    if search_results:
        context += f"\n\nLatest Market Information:\n{search_results}"
    if memory:
        context += f"\n\nConversation Context: {memory}"

    prompt = f"""You are a market analyst as of {CURRENT_DATE}.

{context}

User Query: {query}

Analyze current market conditions, sector performance, economic indicators, 
investment opportunities, and provide short-term and long-term outlook."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return f"📈 **Market Trends Analysis** (as of {CURRENT_DATE})\n\n{response.content}"

def investment_advice_node(state: AgentState) -> AgentState:
    query = state["messages"][-1].content
    response = investment_advice_tool(query, state.get("search_results", ""), state.get("memory_summary", ""))
    state["messages"].append(AIMessage(content=response))
    return state

def portfolio_analysis_node(state: AgentState) -> AgentState:
    query = state["messages"][-1].content
    response = portfolio_analysis_tool(query, state.get("search_results", ""), state.get("memory_summary", ""))
    state["messages"].append(AIMessage(content=response))
    return state

def market_trends_node(state: AgentState) -> AgentState:
    query = state["messages"][-1].content
    response = market_trends_tool(query, state.get("search_results", ""), state.get("memory_summary", ""))
    state["messages"].append(AIMessage(content=response))
    return state

def memory_node(state: AgentState) -> AgentState:
    """Updates conversation memory"""
    if len(state["messages"]) > 6:
        messages_text = "\n".join([f"{m.type}: {m.content[:100]}..." for m in state["messages"][-6:]])
        summary_prompt = f"""Summarize key points in 2-3 sentences: {messages_text}
        Focus on: investment preferences, risk tolerance, topics discussed, user goals."""

        summary = llm.invoke([HumanMessage(content=summary_prompt)])
        state["memory_summary"] = summary.content
    return state

def should_search(state: AgentState) -> str:
    return "search" if state.get("needs_current_data", False) else state["current_tool"]

def route_to_tool(state: AgentState) -> str:
    return state["current_tool"]

# Build graph (cached)
@st.cache_resource
def build_graph():
    workflow = StateGraph(AgentState)

    # Add all nodes
    workflow.add_node("router", router_node)
    workflow.add_node("search", search_node)
    workflow.add_node("investment_advice", investment_advice_node)
    workflow.add_node("portfolio_analysis", portfolio_analysis_node)
    workflow.add_node("market_trends", market_trends_node)
    workflow.add_node("memory", memory_node)

    # Set entry point
    workflow.set_entry_point("router")

    # Router decides: search or go directly to tool
    workflow.add_conditional_edges(
        "router",
        should_search,
        {
            "search": "search",
            "investment_advice": "investment_advice",
            "portfolio_analysis": "portfolio_analysis",
            "market_trends": "market_trends"
        }
    )

    # After search, route to appropriate tool
    workflow.add_conditional_edges(
        "search",
        route_to_tool,
        {
            "investment_advice": "investment_advice",
            "portfolio_analysis": "portfolio_analysis",
            "market_trends": "market_trends"
        }
    )

    # All tools go to memory
    workflow.add_edge("investment_advice", "memory")
    workflow.add_edge("portfolio_analysis", "memory")
    workflow.add_edge("market_trends", "memory")

    # Memory ends the workflow
    workflow.add_edge("memory", END)

    return workflow.compile()

app = build_graph()

# Chat function
def chat(user_input: str):
    """Process user input"""
    try:
        if st.session_state.chat_state is None:
            st.session_state.chat_state = {
                "messages": [SystemMessage(content=f"You are a helpful financial advisor. Today is {CURRENT_DATE}.")],
                "current_tool": "",
                "memory_summary": "",
                "search_results": "",
                "needs_current_data": False
            }

        # Add user message
        st.session_state.chat_state["messages"].append(HumanMessage(content=user_input))

        # Run the graph
        result = app.invoke(st.session_state.chat_state)

        # Update state
        st.session_state.chat_state = result

        # Get response
        response = result["messages"][-1].content
        return response

    except Exception as e:
        error_msg = f"❌ Error processing request: {str(e)}"
        st.error(error_msg)
        return error_msg

# Sidebar
with st.sidebar:
    st.title("💼 AI Financial Advisor")
    st.markdown(f"**Current Date:** {CURRENT_DATE}")
    st.markdown("---")

    st.subheader("📊 Session Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Queries", st.session_state.total_queries)
    with col2:
        st.metric("Messages", len(st.session_state.messages))

    st.markdown("---")

    st.subheader("🔧 Tools Usage")
    for tool, count in st.session_state.tools_used.items():
        icon = {"investment_advice": "💡", "portfolio_analysis": "📊", "market_trends": "📈"}[tool]
        st.markdown(f"{icon} **{tool.replace('_', ' ').title()}**: {count}")

    st.markdown("---")

    st.subheader("💡 Quick Tips")
    with st.expander("What can I ask?"):
        st.markdown("""
        - **Investment Advice**: "Should I invest in ETFs?"
        - **Portfolio Analysis**: "How should I allocate my portfolio?"
        - **Market Trends**: "What are current tech stock trends?"
        - Add "current" or "latest" for real-time data
        """)

    with st.expander("Features"):
        st.markdown("""
        - ✅ Real-time market data
        - ✅ Conversation memory
        - ✅ Intelligent routing
        - ✅ 3 specialized tools
        """)

    st.markdown("---")

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_state = None
        st.session_state.total_queries = 0
        st.session_state.tools_used = {"investment_advice": 0, "portfolio_analysis": 0, "market_trends": 0}
        st.rerun()

# Main content
st.title("🤖 AI Financial Investment Advisor")
st.markdown("Ask me anything about investments, portfolio allocation, or market trends!")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about investments, portfolios, or market trends..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Get bot response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🤔 Thinking..."):
            response = chat(prompt)
        st.markdown(response)

    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.total_queries += 1

    # Force rerun to update sidebar stats
    st.rerun()

# Welcome message
if len(st.session_state.messages) == 0:
    st.info("👋 Welcome! I'm your AI Financial Advisor. Ask me anything about investments, portfolio management, or market trends. I can provide up-to-date information as of November 24, 2025!")

    # Example queries
    st.markdown("### 💭 Try asking:")
    col1, col2, col3 = st.columns(3)

    example_clicked = False
    query_to_process = None

    with col1:
        if st.button("📈 Current market trends", use_container_width=True):
            query_to_process = "What are the current market trends?"
            example_clicked = True

    with col2:
        if st.button("💼 Portfolio allocation", use_container_width=True):
            query_to_process = "How should I allocate my portfolio with moderate risk?"
            example_clicked = True

    with col3:
        if st.button("💡 Investment advice", use_container_width=True):
            query_to_process = "What are the best investments for beginners?"
            example_clicked = True

    # Handle example query clicks
    if example_clicked and query_to_process:
        st.session_state.messages.append({"role": "user", "content": query_to_process})
        with st.chat_message("user", avatar="👤"):
            st.markdown(query_to_process)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🤔 Thinking..."):
                response = chat(query_to_process)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.total_queries += 1
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; padding: 1rem;'>
    <p>💼 AI Financial Advisor | Powered by LangGraph & ChatOpenAI | Data current as of November 24, 2025</p>
    <p style='font-size: 0.8rem; opacity: 0.7;'>Disclaimer: This is for educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)