import {NextResponse} from 'next/server'

import OpenAI from 'openai'
import { Pinecone } from '@pinecone-database/pinecone'

const systemPrompt=
`You are a helpful and knowledgeable assistant designed to help students find the best professors based on their specific queries. Using Retrieval-Augmented Generation (RAG), you will analyze the user’s question and provide the top 3 professors who best match their criteria. Each professor recommendation should include the professor's name, the subject they teach, their rating, and a brief summary of why they are a good match based on the query.

When providing the recommendations, ensure the information is accurate, clear, and concise. If the user query is vague, use your best judgment to interpret and return the most relevant results.
`

export async function POST(req) {
    const data = await req.json()
    const pc = new Pinecone({
        apikey: process.env.PINECONE_API_KEY,
    })
    const index = pc.index('rag').namespace('ns1')
    const openai = new OpenAI()

    const text = data[data.length - 1].content
    const embedding = await openai.embeddings.create({
        model: 'text-embedding-3-small',
        input: 'text',
        encoding_format: 'float',
    })

    const results = await index.query({
        topK: 5,
        includeMetadata: true,
        vector: embedding.data[0].embedding,
    })

    let resultString = 'Returned reesut from vector db (done automatically): '
    results.matches.forEach((match) => {
        resultString += `
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
