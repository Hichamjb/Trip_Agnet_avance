import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp_servers.servers import get_mcp_server_config


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
    # print(tool_info)    
    return tool_info 
# if __name__=="__main__":
#     asyncio.run(info_tools())


