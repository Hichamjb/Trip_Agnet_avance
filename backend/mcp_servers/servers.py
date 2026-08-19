# mcp/servers.py

import os
from config.config import config


def get_mcp_server_config() -> dict:
  
    servers = {
        # 1. Tavily Web Search MCP
        "tavily-mcp": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "tavily-mcp@latest"],
            "env": {
                "TAVILY_API_KEY": config.TAVILY_API_KEY,
                "DEFAULT_PARAMETERS": '{"include_images": true, "max_results": 15, "search_depth": "advanced"}',
                "PATH": os.getenv("PATH", ""),
            },
        },
        # 2. Weather MCP
        "weather": {
            "transport": "stdio",
            "command": "mcp-server-weather", 
            "args":  ["--api_key", config.WEATHER_API_KEY],
            "env": {
                
                "PATH": os.getenv("PATH", ""),
            },
        },
       
        "web_fetch": {
            "transport": "stdio",
            "command": "python",
            "args": ["-m", "mcp_server_fetch"],
            "env": {"PATH": os.getenv("PATH", "")},
        },
    }

    return servers