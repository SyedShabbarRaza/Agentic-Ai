from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv()

# Initialize LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("Groq_Api_Key") # Standardizing naming conventions
)

@tool
def get_weather(city: str) -> str:
    """Get current weather for a specific city."""
    weather_data = {
        "lahore": "40 degrees. Very hot outside.",
        "karachi": "35 degrees. Very hot outside.",
        "islamabad": "29 degrees. Normal outside." # Fixed your 299 degrees typo!
    }
    # Safely convert to lowercase to handle any capitalizations from the LLM
    return weather_data.get(city.lower(), f"Sorry, data for {city} is not available.")

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))  

# Bind tools for the initial detection phase
llm_with_tools = llm.bind_tools([get_weather, calculator])

# 1. User asks the question
user_prompt = "What is current weather in lahore? and what is 2+2?"
messages = [HumanMessage(content=user_prompt)]

# 2. First invocation: LLM decides to call the tool
response = llm_with_tools.invoke(messages)
messages.append(response)

# 3. Execute the tool if the model requested it
if response.tool_calls:
    tool_call = response.tool_calls[0]
    
    # Run the tool with the arguments the model extracted
    if tool_call["name"] == "get_weather":
        result = get_weather.invoke(tool_call["args"])
    elif tool_call["name"] == "calculator":
        result = calculator.invoke(tool_call["args"])
    
    # Append the tool result back to the conversation chain
    messages.append(
        ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        )
    )

# 4. Final invocation: Use the BASE llm (not llm_with_tools) to answer the user
final_response = llm.invoke(messages)

print("\n--- Final Answer from AI ---")
print(final_response.content)
