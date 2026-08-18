from typing import List, Tuple, Any,TypedDict,Annotated
from langchain_core.messages import (
    AnyMessage
 
)
import operator

class State(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]