import {NextResponse} from 'next/server'

import OpenAI from 'openai'
import { Pinecone } from '@pinecone-database/pinecone'

const systemPrompt = `
  You are a friendly and knowledgeable assistant designed to help students find the best professors based on their specific queries. 

  **Your main task is to provide professor recommendations** using Retrieval-Augmented Generation (RAG). For each query, analyze the user's question and provide the top 3 professors who best match their criteria. Include the following details in your response:
  - **Professor's Name**
  - **Subject They Teach**
  - **Rating**
  - **Brief Summary** of why they are a good match based on the query. Ensure the information is accurate, clear, and concise.

  **Handling General Conversational Inputs:**
  - If the user expresses gratitude with phrases like "thank you" or "thanks," respond with a polite acknowledgment. For example: "You're welcome! If you have any more questions or need further assistance, feel free to ask."
  - If the user asks a question or makes a request not related to professor recommendations, provide a relevant and helpful response or guide them on how to proceed.
  

  **For Vague Queries:**
  - If the user’s query is vague or unclear, use your best judgment to interpret their needs and provide the most relevant professor recommendations. If needed, ask clarifying questions to better understand the user’s requirements.
`;


export async function POST(req) {
    const data = await req.json()
    const pc = new Pinecone({
     apiKey: process.env.PINECONE_API_KEY,
    })
    const index = pc.index('rag').namespace('ns1')
    const openai = new OpenAI()

    const text = data[data.length - 1].content
    const embedding = await openai.embeddings.create({
        model: 'text-embedding-3-small',
        input: text,
        encoding_format: 'float',
    })

    const results = await index.query({
        topK: 5,
        includeMetadata: true,
        vector: embedding.data[0].embedding
    })

    let resultString = 'Returned result from vector db (done automatically): '
    results.matches.forEach((match) => {
        resultString += `\n
        Returned Results:
        Professor: ${match.id}
        Review: ${match.metadata.stars}
        Subject: ${match.metadata.subject}
        Stars: ${match.metadata.stars}
        \n\n
        `
    })

    const lastMessage = data[data.length - 1]
    const lastMessageContent = lastMessage.content + resultString
    const lastDataWithoutLastMessage = data.slice(0, data.length - 1)

    const completion = await openai.chat.completions.create({
        messages: [
          {role: 'system', content: systemPrompt},
          ...lastDataWithoutLastMessage,
          {role: 'user', content: lastMessageContent},
        ],
        model: 'gpt-3.5-turbo',
        stream: true,
      })

      const stream = new ReadableStream({
        async start(controller) {
          const encoder = new TextEncoder()
          try {
            for await (const chunk of completion) {
              const content = chunk.choices[0]?.delta?.content
              if (content) {
                const text = encoder.encode(content)
                controller.enqueue(text)
              }
            }
          } catch (err) {
            controller.error(err)
          } finally {
            controller.close()
          }
        },
      })
      return new NextResponse(stream)

}
