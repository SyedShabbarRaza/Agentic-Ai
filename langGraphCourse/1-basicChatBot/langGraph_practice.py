import os
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode, tools_condition
load_dotenv()
from langchain_groq import ChatGroq
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph import START, StateGraph, add_messages

# 1. CAllING LLM FOR A QUESTION

llm=ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("Groq_Api_Key")
)

# response=llm.invoke("What is langgraph?")
# print(response.content)


# 2. FOR LANGGRAPH -> STATE FOR STORING DATA
#LIKE IN REACT WE HAVE CONST [STATE,SETSTATE]=USESTATE()

class State(TypedDict):
    messages: Annotated[list,add_messages]


# Let's make tools

def multiply(a:int,b:int)->int:
    """Multiplies two integers."""
    return a * b

def get_weather(city:str)->str:
    """Returns the weather for a given city."""
    # For demonstration purposes, we'll return a dummy weather.
    return f"The weather in {city} is sunny."


llm_with_tools=llm.bind_tools([multiply, get_weather])


# 3. Let's Go Towrds llm Calling

# IT'S ALSO A NODE
def llm_with_tool_calling(state:State):
    return {"messages":[llm_with_tools.invoke(state["messages"])]}

# langGraph starting WORKFLOW
# ADD_NODES
builder=StateGraph(State)
builder.add_node("llm_with_tool_calling",llm_with_tool_calling)
builder.add_node("tools",ToolNode([multiply,get_weather]))

# ADD_EDGES
builder.add_edge(START,"llm_with_tool_calling")
builder.add_conditional_edges("llm_with_tool_calling",
tools_condition 
#llm_with_tool_calling sent response aggar 
# tool use krna to go forward otherwise yaha hi END 
)
builder.add_edge("tools",
"llm_with_tool_calling"
# Ab jab response tool sy aya hai to usy wapis llm ko bhej do so that
#  wo khud decide kry END krna ya abi or aagy jana
)

graph=builder.compile()

# RUN
response = graph.invoke({"messages": [("user", "Give me weather of lahore using tool and also 2*100")]})
print(response)


# Formated Response
# for msg in response['messages']:
#     msg.pretty_print()