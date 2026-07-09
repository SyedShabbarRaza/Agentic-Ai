from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages:Annotated[list,add_messages]




import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("Groq_Api_Key")
)

# def chatbot(state:State):
#     return {"messages":[llm.invoke(state["messages"])]}

# graph_builder=StateGraph(State)


# graph_builder.add_node("llmchatbot",chatbot)

# graph_builder.add_edge(START,"llmchatbot")
# graph_builder.add_edge("llmchatbot",END)


# graph=graph_builder.compile()

# response=graph.invoke({"messages":"Hi"})
# print(response)
# response["messages"][-1].content



from langchain_tavily import TavilySearch

tool=TavilySearch(max_results=2)
response=tool.invoke("What is langgraph")
# print(response)

## Custom function
def multiply(a:int,b:int)->int:
    """Multiply a and b

    Args:
        a (int): first int
        b (int): second int

    Returns:
        int: output int
    """
    return a*b

tools=[tool,multiply]

llm_with_tool=llm.bind_tools(tools)

# print(llm_with_tool)



#Stategraph without ReAct Agent

# from langgraph.prebuilt import ToolNode
# from langgraph.prebuilt import tools_condition

# ## Node definition
# def tool_calling_llm(state:State):
#     return {"messages":[llm_with_tool.invoke(state["messages"])]}

# ## Grpah
# builder=StateGraph(State)
# builder.add_node("tool_calling_llm",tool_calling_llm)
# builder.add_node("tools",ToolNode(tools))

# ## Add Edges
# builder.add_edge(START, "tool_calling_llm")
# builder.add_conditional_edges(
#     "tool_calling_llm",
#     # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
#     # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
#     tools_condition
# )
# builder.add_edge("tools",END)

# ## compile the graph
# graph=builder.compile()

# response=graph.invoke({"messages":"What is the recent ai news"})
# print(response)
# # response['messages'][-1].content


#Stategraph with ReAct Agent

from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

## Node definition
def tool_calling_llm(state:State):
    return {"messages":[llm_with_tool.invoke(state["messages"])]}

## Grpah
builder=StateGraph(State)
builder.add_node("tool_calling_llm",tool_calling_llm)
builder.add_node("tools",ToolNode(tools))

## Add Edges
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition
)
# builder.add_edge("tools",END) End krny ky bjaye wapis LLM ko kaho decide kry koi or tool use krny ki zaroorat ti nahi na 
builder.add_edge("tools", "tool_calling_llm")

## compile the graph
graph=builder.compile()

response=graph.invoke({"messages":"Give me the recent ai news and then multiply 5 by 10"})
# print(response)
# Achy sy print krwany ky liye
for m in response['messages']:
    m.pretty_print()
# response['messages'][-1].content

