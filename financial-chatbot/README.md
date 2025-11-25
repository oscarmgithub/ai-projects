

User Input
    ↓
┌─────────────┐
│   Router    │  (Analyzes query & decides path)
└─────────────┘
    ↓
    ├─── Needs Current Data? ───┐
    │                            │
   YES                          NO
    │                            │
    ↓                            ↓
┌─────────────┐           ┌─────────────┐
│   Search    │           │  Direct to  │
│    Node     │           │    Tool     │
└─────────────┘           └─────────────┘
    ↓                            │
    └──────────┬─────────────────┘
               ↓
      ┌────────────────┐
      │  Tool Selection│
      └────────────────┘
               ↓
    ┌──────────┴──────────┐
    │                     │
┌─────────┐  ┌──────────┐  ┌─────────┐
│Investment│  │Portfolio │  │ Market  │
│ Advice  │  │ Analysis │  │ Trends  │
└─────────┘  └──────────┘  └─────────┘
    │            │              │
    └────────────┼──────────────┘
                 ↓
          ┌─────────────┐
          │   Memory    │  (Updates context)
          └─────────────┘
                 ↓
            Response



Data Flow Visualization
┌─────────────────────────────────────────────────────────┐
│                      User Input                         │
└───────────────────────┬─────────────────────────────────┘
                        ↓
                   [State Created]
                 messages: [UserMsg]
                 current_tool: ""
                 memory_summary: ""
                 search_results: ""
                 needs_current_data: False
                        ↓
                  ┌──────────┐
                  │  Router  │
                  └──────────┘
                        ↓
           [State Updated with Routing Info]
                 current_tool: "market_trends"
                 needs_current_data: True
                        ↓
               ┌────────┴────────┐
               │   Conditional   │
               │  should_search  │
               └────────┬────────┘
                        ↓
                  ┌─────────┐
                  │ Search  │
                  └─────────┘
                        ↓
            [State Enhanced with Search]
                 search_results: "NVIDIA up 5%..."
                        ↓
               ┌────────┴────────┐
               │   Conditional   │
               │  route_to_tool  │
               └────────┬────────┘
                        ↓
              ┌──────────────────┐
              │  Market Trends   │
              │      Tool        │
              └──────────────────┘
                        ↓
          [State Enhanced with Response]
                 messages: [UserMsg, AIMsg]
                        ↓
                  ┌─────────┐
                  │ Memory  │
                  └─────────┘
                        ↓
          [State Enhanced with Summary]
                 memory_summary: "User interested in tech..."
                        ↓
                      [END]
                        ↓
┌─────────────────────────────────────────────────────────┐
│              Response Returned to User                  │
└─────────────────────────────────────────────────────────┘


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
