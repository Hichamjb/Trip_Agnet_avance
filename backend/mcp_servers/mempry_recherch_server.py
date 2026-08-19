from pathlib import Path
from typing import List
from dotenv import load_dotenv
import os
import certifi

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
from fastmcp  import FastMCP
from backend.rag.rag import retrieve_from_rag
mcp = FastMCP()
@mcp
def cherch_in_rag(query: str, thread_id: str, k: int = 4):
    """_summary_

    Args:
        query (str): _description_
        thread_id (str): _description_
        k (int, optional): _description_. Defaults to 4.

    Returns:
        _type_: _description_
    """
    return retrieve_from_rag(query,int)
