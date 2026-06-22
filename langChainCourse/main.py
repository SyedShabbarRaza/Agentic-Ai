import os
import json
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

client=Groq(api_key= os.getenv("Groq_Api_Key"))

def get_weather(city):
    weather_data={
        "lahore":"39 degrees. Very hot outside.",
        "karachi":"35 degrees. Very hot outside.",
        "islamabad":"299 degrees. normal outside."
    }
    return weather_data.get(city.lower(),
     f"Sorry, data of {city} is not available."
     )

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name"
                    }
                },
                "required": ["city"]
            }
        }
    }
]



messages = []

while True:
    question = input("You: ")

    if question.lower() == "exit":
        break

    messages.append({
        "role": "user",
        "content": question
    })

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        tools=tools,
    )

    message = completion.choices[0].message

    if message.tool_calls:
        messages.append(message)
        tool_call =message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)
        city = args["city"]
        result = get_weather(city)
        # print("Tool Result:", result)

        messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": result
        })

        final_completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
        )

        final_answer = final_completion.choices[0].message.content
        print("AI:", final_answer)

    else:
        print("AI:", message.content)

    messages.append({
        "role": "assistant",
        "content": message.content
    })


# Dummy weather logic to understand LLMs
# while True:
#     question=input ("You: ")

#     if "weather" in question.lower():
#         if "lahore" in question.lower():
#             weather=get_weather("lahore")
#         elif "karachi" in question.lower():
#             weather=get_weather("karachi")
#         elif "islamabad" in question.lower():
#             weather=get_weather("islamabad")
#         else:
#             print("Sorry, I don't have weather data for that city.")

#         print("AI:",weather)