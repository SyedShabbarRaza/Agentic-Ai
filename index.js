// const OpenAI = require('openai');
// require('dotenv').config();
import Groq from 'groq-sdk';
import dotenv from 'dotenv';
dotenv.config();

const groq = new Groq({
  apiKey: process.env.Groq_Api_Key,
});

async function main() {
  try {
    const completion = await groq.chat.completions.create({
        model: "llama-3.3-70b-versatile",
      messages: [
        { role: "user", content: "What is the weather in lahore today?" },
  { role: "assistant", content: "Hi!" },
  { role: "user", content: "What did I just say?" },
  { role: "user", content: "2+2=?" }
      ],
    });

    console.log(completion.choices[0]?.message?.content);
  } catch (error) {
    console.error("Error:", error);
  }
}

main();