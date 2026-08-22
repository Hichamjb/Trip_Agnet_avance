import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from ..mcp_servers.servers import get_mcp_server_config


async def create_tools():
    server = get_mcp_server_config()
    client = MultiServerMCPClient(server)
    tools = await client.get_tools()
    return tools


async def info_tools():
    tools = await create_tools()
    tool_info = []
    for tool in tools:
        tool_info.append({
            "name": tool.name,
            "description": tool.description
        })
     
    return tool_info 




async def create_tools_memory():
    client2 = M= MultiServerMCPClient(
    {
        "Long_memory": {
            "transport": "sse",
            "url": "http://127.0.0.1:8001/sse",
        }
    }
    )
    tools = await client2.get_tools()
    return tools