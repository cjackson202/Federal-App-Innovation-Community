'''

This code was referenced from https://github.com/Azure-Samples/AI-Gateway/blob/main/labs/model-context-protocol/model-context-protocol.ipynb

The script connects to an MCP tool server, lists available tools, and uses Azure OpenAI to determine which tools to invoke based on a given prompt. 
It then calls the appropriate tools and completes the response using the tool outputs.

**Make sure to replace**:
- Lines 64 & 100 with the name of your deployed model
- Line 122 with a relevant query to your Azure AI Search Index 

'''
import json
import asyncio
import os
from mcp import ClientSession
from mcp.client.sse import sse_client
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

async def call_tool(mcp_session, function_name, function_args):
    try:
        func_response = await mcp_session.call_tool(function_name, function_args)
        func_response_content = func_response.content
    except Exception as e:
        func_response_content = json.dumps({"error": str(e)})
    return str(func_response_content)

async def run_completion_with_tools(server_url, prompt):
    streams = None
    session = None
    try:
        # Connect to MCP tool server
        streams_ctx = sse_client(server_url)
        streams = await streams_ctx.__aenter__()
        session_ctx = ClientSession(streams[0], streams[1])
        session = await session_ctx.__aenter__()
        await session.initialize()
        response = await session.list_tools()
        tools = response.tools
        print(f"✅ Connected to server {server_url}")

        # Build OpenAI-compatible tools spec
        openai_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "parameters": tool.inputSchema
            },
        } for tool in tools]

        # Step 1: Ask the model which function to call
        print("▶️ Step 1: start a completion to identify the appropriate functions to invoke based on the prompt")
        client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version=os.getenv("AZURE_OPENAI_VERSION"),
        )

        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model="your_deployed_model_name",  # USE your deployment name here
            messages=messages,
            tools=openai_tools,
            store=True,
            metadata={
                "user": "admin",
                "category": "mcp-test",
            }
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        print(f"Tools from step 1: response: {response_message}, tool_calls: {tool_calls}\n")

        if tool_calls:
            # Step 2: Call the tools
            messages.append(response_message)
            print("▶️ Step 2: call the functions")
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                print(f"   Function Name: {function_name} | Args: {function_args}")

                function_response = await call_tool(session, function_name, function_args)
                print(f"   Function Response: {function_response}")

                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                })

            # Step 3: Let the model finish the response
            print("▶️ Step 3: finish with a completion to answer the user prompt using the function response")
            second_response = client.chat.completions.create(
                model="your_deployed_model_name",  # USE your deployment name here
                messages=messages,
                store=True,
                metadata={
                    "user": "admin",
                    "category": "mcp-test",
                }
            )
            print("💬", second_response.choices[0].message.content)
        else:
            print("⚠️ No tools were called by the model.")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if session:
            await session_ctx.__aexit__(None, None, None)
        if streams:
            await streams_ctx.__aexit__(None, None, None)

# Run the workflow
asyncio.run(run_completion_with_tools("http://0.0.0.0:8080/sse", "<Please place search query here relevant to the Azure AI Search Index>"))
