"""
Streamlit UI for Multi-Agent Financial News System
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_tavily import TavilySearch
import os
from datetime import datetime
import time
from dotenv import load_dotenv

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI News Research System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .agent-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid;
    }
    .research-card {
        background-color: #e3f2fd;
        border-left-color: #2196F3;
    }
    .editor-card {
        background-color: #fff3e0;
        border-left-color: #FF9800;
    }
    .writer-card {
        background-color: #e8f5e9;
        border-left-color: #4CAF50;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'research_results' not in st.session_state:
    st.session_state.research_results = None
if 'editor_feedback' not in st.session_state:
    st.session_state.editor_feedback = None
if 'final_article' not in st.session_state:
    st.session_state.final_article = None
if 'is_running' not in st.session_state:
    st.session_state.is_running = False

# Define the state
class AgentState(TypedDict):
    research_results: str
    editor_feedback: str
    final_article: str
    messages: Annotated[list, operator.add]

# Initialize LLM and Search (only once)
@st.cache_resource
def initialize_tools():
    llm = ChatOpenAI(
        model=st.session_state.get('model', 'gpt-4'),
        api_key=st.session_state.get('openai_key', os.getenv("OPENAI_API_KEY")),
        temperature=0.7
    )

    tavily_search = TavilySearch(
        api_key=st.session_state.get('tavily_key', os.getenv("TAVILY_API_KEY")),
        max_results=5
    )

    return llm, tavily_search

def search_financial_news(query: str, tavily_search) -> str:
    """Search for current financial news using Tavily"""
    try:
        response = tavily_search.invoke({"query": query})
        results = response.get('results', [])

        if not results:
            return "No results found for this query."

        formatted_results = []
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            url = result.get('url', 'No URL')
            content = result.get('content', 'No content')

            formatted_results.append(
                f"{i}. {title}\n"
                f"   Source: {url}\n"
                f"   Content: {content}\n"
            )

        return "\n".join(formatted_results)

    except Exception as e:
        return f"Error searching '{query}': {str(e)}"

# Agent 1: Research Agent
def research_agent(state: AgentState, llm, tavily_search, progress_bar, status_text) -> AgentState:
    today = datetime.now().strftime("%B %d, %Y")

    status_text.text("🔍 Research Agent: Searching for Mag 7 stocks news...")
    progress_bar.progress(0.1)

    queries = [
        f"Magnificent 7 stocks news today {today}",
        f"Apple Microsoft Google Amazon Meta Tesla Nvidia stock news {today}",
        f"mag 7 tech stocks performance {today} CNBC Bloomberg",
        f"big tech earnings magnificent seven {today}"
    ]

    all_results = []
    for i, query in enumerate(queries):
        status_text.text(f"🔍 Searching: {query}")
        results = search_financial_news(query, tavily_search)
        all_results.append(results)
        progress_bar.progress(0.1 + (i + 1) * 0.1)
        time.sleep(0.5)

    combined_search_results = "\n\n".join(all_results)

    system_prompt = f"""You are a financial research analyst specializing in finding 
    breaking news that affects the Magnificent 7 (Mag 7) stocks. Today is {today}.
    
    The Magnificent 7 stocks are:
    1. Apple (AAPL)
    2. Microsoft (MSFT)
    3. Alphabet/Google (GOOGL)
    4. Amazon (AMZN)
    5. Meta/Facebook (META)
    6. Tesla (TSLA)
    7. Nvidia (NVDA)
    
    Your job is to:
    1. Analyze the search results provided about these stocks
    2. Identify the most important news from TODAY affecting any of these companies
    3. Focus on news from reputable sources like CNBC, Bloomberg, Reuters, WSJ
    4. Highlight potential market impact for each company mentioned
    5. Provide 3-5 key news items with brief summaries and sources
    
    Format your findings clearly with sources and potential market implications for the Mag 7."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""Based on these search results from TODAY ({today}), 
        summarize the most important news affecting the Magnificent 7 stocks:
        
        SEARCH RESULTS:
        {combined_search_results}
        
        Provide a clear summary of the top 3-5 stories about the Mag 7 companies (Apple, Microsoft, Google, Amazon, Meta, Tesla, Nvidia) with their sources and market implications.""")
    ]

    status_text.text("🔍 Research Agent: Analyzing results...")
    progress_bar.progress(0.5)
    response = llm.invoke(messages)

    return {
        **state,
        "research_results": response.content,
        "messages": [("research_agent", response.content)]
    }

# Agent 2: Chief Editor Agent
def editor_agent(state: AgentState, llm, progress_bar, status_text) -> AgentState:
    status_text.text("✏️ Chief Editor: Reviewing research findings...")
    progress_bar.progress(0.6)

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
    progress_bar.progress(0.7)

    return {
        **state,
        "editor_feedback": response.content,
        "messages": [("editor_agent", response.content)]
    }

# Agent 3: Writer Agent
def writer_agent(state: AgentState, llm, progress_bar, status_text) -> AgentState:
    status_text.text("📝 Writer Agent: Crafting final article...")
    progress_bar.progress(0.8)

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
    progress_bar.progress(1.0)

    return {
        **state,
        "final_article": response.content,
        "messages": [("writer_agent", response.content)]
    }

# Build the workflow
def create_workflow(llm, tavily_search, progress_bar, status_text):
    workflow = StateGraph(AgentState)

    workflow.add_node("research", lambda state: research_agent(state, llm, tavily_search, progress_bar, status_text))
    workflow.add_node("editor", lambda state: editor_agent(state, llm, progress_bar, status_text))
    workflow.add_node("writer", lambda state: writer_agent(state, llm, progress_bar, status_text))

    workflow.set_entry_point("research")
    workflow.add_edge("research", "editor")
    workflow.add_edge("editor", "writer")
    workflow.add_edge("writer", END)

    return workflow.compile()

# Main UI
def main():
    st.markdown('<h1 class="main-header">📰 Mag 7 Stocks News System</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Research on the Magnificent 7 Tech Stocks")

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        openai_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        tavily_key = st.text_input("Tavily API Key", type="password", value=os.getenv("TAVILY_API_KEY", ""))

        model_choice = st.selectbox(
            "Select Model",
            ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
            index=0
        )

        st.session_state.openai_key = openai_key
        st.session_state.tavily_key = tavily_key
        st.session_state.model = model_choice

        st.divider()
        st.markdown("### 🤖 Agent Pipeline")
        st.markdown("""
        1. **Research Agent** 🔍  
           Searches for Mag 7 stocks news  
           (AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA)
        
        2. **Chief Editor** ✏️  
           Reviews & validates findings
        
        3. **Writer Agent** 📝  
           Creates final article
        """)

        st.divider()
        st.markdown("### 📊 About")
        st.info("This system uses LangGraph to orchestrate multiple AI agents that research, review, and write financial news articles about the Magnificent 7 stocks (Apple, Microsoft, Google, Amazon, Meta, Tesla, Nvidia) based on real-time data.")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col2:
        st.markdown("### 🎯 Quick Info")
        st.info(f"📅 **Date**: {datetime.now().strftime('%B %d, %Y')}")

        if st.button("🚀 Generate Mag 7 News Article", disabled=st.session_state.is_running):
            if not openai_key or not tavily_key:
                st.error("⚠️ Please provide both API keys in the sidebar!")
            else:
                st.session_state.is_running = True
                st.rerun()

    with col1:
        if st.session_state.is_running:
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                llm, tavily_search = initialize_tools()
                app = create_workflow(llm, tavily_search, progress_bar, status_text)

                initial_state = {
                    "research_results": "",
                    "editor_feedback": "",
                    "final_article": "",
                    "messages": []
                }

                result = app.invoke(initial_state)

                st.session_state.research_results = result["research_results"]
                st.session_state.editor_feedback = result["editor_feedback"]
                st.session_state.final_article = result["final_article"]

                status_text.text("✅ Complete!")
                time.sleep(1)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
            finally:
                st.session_state.is_running = False
                st.rerun()

        # Display results
        if st.session_state.final_article:
            st.success("✅ Article Generation Complete!")

            # Tabs for different outputs
            tab1, tab2, tab3 = st.tabs(["📰 Final Article", "🔍 Research", "✏️ Editor Review"])

            with tab1:
                st.markdown('<div class="agent-card writer-card">', unsafe_allow_html=True)
                st.markdown("### 📝 Final Article")
                st.markdown(st.session_state.final_article)
                st.markdown('</div>', unsafe_allow_html=True)

                # Download button
                st.download_button(
                    label="📥 Download Article",
                    data=st.session_state.final_article,
                    file_name=f"market_news_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )

            with tab2:
                st.markdown('<div class="agent-card research-card">', unsafe_allow_html=True)
                st.markdown("### 🔍 Research Findings")
                st.markdown(st.session_state.research_results)
                st.markdown('</div>', unsafe_allow_html=True)

            with tab3:
                st.markdown('<div class="agent-card editor-card">', unsafe_allow_html=True)
                st.markdown("### ✏️ Editorial Feedback")
                st.markdown(st.session_state.editor_feedback)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("👈 Click 'Generate Mag 7 News Article' to start the AI pipeline!")

            # Demo section
            with st.expander("ℹ️ How it works"):
                st.markdown("""
                ### Multi-Agent Workflow for Mag 7 Stocks
                
                **The Magnificent 7 are:**
                - 🍎 Apple (AAPL)
                - 💻 Microsoft (MSFT)
                - 🔍 Alphabet/Google (GOOGL)
                - 📦 Amazon (AMZN)
                - 👥 Meta/Facebook (META)
                - 🚗 Tesla (TSLA)
                - 🎮 Nvidia (NVDA)
                
                **Process:**
                1. **Research Agent** searches multiple financial news sources for Mag 7 stock updates
                2. **Chief Editor** reviews findings for accuracy and provides guidance
                3. **Writer Agent** crafts a professional article based on research and editorial feedback
                
                All using real-time web data and GPT-4!
                """)

if __name__ == "__main__":
    main()