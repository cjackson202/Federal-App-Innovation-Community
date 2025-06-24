'''

This code was referenced from https://github.com/Azure-Samples/AI-Gateway/blob/main/labs/model-context-protocol/model-context-protocol.ipynb


The script sets up an AI agent using Azure OpenAI and MCP SSE Plugin to respond to user queries based on a given prompt. 
It demonstrates how to create an agent, sync with your MCP cilent, invoke it for a response, and clean up the session.

**Make sure to replace**:
- 'your_deployed_model_name' with the name of your deployed model on line 33.
- 'Please place search query here relevant to the Azure AI Search Index>' with a relevant query to your Azure AI Search Index 
    on line 24

'''

import asyncio
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.mcp import MCPSsePlugin
import os
from dotenv import load_dotenv
load_dotenv()

user_input = "<Please place search query here relevant to the Azure AI Search Index>"

async def main():
    # 1. Create the agent
    async with MCPSsePlugin(
        name="Weather",
        url=f"http://127.0.0.1:8080/sse",
        description="Azure AI Search Plugin",
    ) as weather_plugin:
        agent = ChatCompletionAgent(
            service=AzureChatCompletion(
                endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_KEY"),
                api_version=os.getenv("AZURE_OPENAI_VERSION"),                
                deployment_name="your_deployed_model_name" # USE your deployment name here
            ),
            name="MCPAgent",
            instructions="Answer questions using the tools provided.",
            plugins=[weather_plugin],
        )

        thread: ChatHistoryAgentThread | None = None

        print(f"# User: {user_input}")
        # 2. Invoke the agent for a response
        response = await agent.get_response(messages=user_input, thread=thread)
        print(f"# {response.name}: {response} ")
        thread = response.thread # type: ignore

        # 3. Cleanup: Clear the thread
        await thread.delete() if thread else None

if __name__ == "__main__":
    asyncio.run(main())