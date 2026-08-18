import json
# from typing import List, Tuple, Any,TypedDict, Annotated
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import (
    AnyMessage,
    HumanMessage
 
)
from psycopg import AsyncConnection
import psycopg
import asyncio
from psycopg.rows import dict_row
#  CORRECT
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from backend.config import config
from backend.tools import create_tools
from backend.state import State
from mcp_servers.servers import get_mcp_server_config
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import uuid


def get_database_url():
    database_url = config.DATABASE_URL

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. Please add your Render PostgreSQL External Database URL to .env"
        )

    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url




SYSTEM_PROMPT = """You are an advanced AI travel planning assistant.

You have access to external MCP tools.

Your responsibilities:
1. Understand the user's travel request.
2. Identify what information is required.
3. Select the appropriate MCP tools.
4. Use multiple tools when necessary.
5. Never invent information returned by external tools.
6. Never invent:
   - flight prices
   - hotel prices
   - restaurant information
   - weather
   - distances
   - opening hours
   - availability
7. For current information, prefer external tools.
8. Carefully analyze tool results before answering.
9. If a tool returns incomplete information, clearly say that the information is incomplete.
10. Distinguish:
    - confirmed information
    - estimated information
11. If the user request is ambiguous, ask for the missing information.
12. When planning a complete trip, organize the answer into:
    - Destination
    - Travel dates
    - Transportation
    - Accommodation
    - Weather
    - Activities
    - Restaurants
    - Estimated budget
    - Important notes
13. When comparing destinations, provide a clear comparison.
14. Do not expose internal tool calls unless the user asks for them.
15. Answer in the same language as the user."""




async def build_trip_agent() :
    """
    Initializes MultiServerMCPClient, discovers tools dynamically, binds them to ChatGroq,
    and builds the LangGraph StateGraph compiled app.
    """
    # 1. Connect directly to MCP servers defined in mcp/servers.py
  
    # 2. Dynamically discover tools from all configured MCP servers
    

  

    # 4. Initialize LLM (ChatGroq llama-3.3-70b-versatile)
    load_dotenv()


# LLM 
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite"
    )
    tools = await create_tools()
    

    # 5. Bind tools directly to the LLM
    llm_with_tools = llm.bind_tools(tools)

    # 6. Define Agent Node
    async def agent_node(state: State) -> dict:
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    # 7. Construct LangGraph Workflow
    workflow = StateGraph(State)

    workflow.add_node("agent", agent_node)

    tool_node = ToolNode(tools)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")
    workflow.add_edge("agent", END)
    # =========================
    # PostgreSQL Checkpointer
    # =========================
    DATABASE_URL = get_database_url()

    

    conn = await AsyncConnection.connect(
        DATABASE_URL,
        autocommit=True,
        row_factory=dict_row
    )
    checkpointer = AsyncPostgresSaver(conn)
    await checkpointer.setup()

    app = workflow.compile(checkpointer=checkpointer)
    return app


#  test
# Exemple avec conservation d'historique simple :

def genre_thread_id():
    thread_id = f"user_{uuid.uuid4().hex}"
    return thread_id
async def test_agent(user_input=None,thread_id=None):
   
    if thread_id is None:
        thread_id = genre_thread_id()

    config_user = {
        "configurable": {
            "thread_id": thread_id
        }
    }  
    agent = await build_trip_agent()
    history = []  # Maintain clean message list

    while True:
        user_input = input("\n enter your question: ")
        if user_input.lower() in ["exit", "stop", "quit"]:
            break

        # Append individual HumanMessage object (not a list or tuple)
        history.append(HumanMessage(content=user_input))

        # Invoke agent with state
        response = await agent.ainvoke({"messages": history},config=config_user)

        # Update history with the returned full message list from state
        history = response["messages"]

        # Print the latest response from the AI
        latest_message = history[-1]
        print(f"\nAssistant: {latest_message.content[0]['text']}") 
if __name__== "__main__":

    asyncio.run(test_agent())