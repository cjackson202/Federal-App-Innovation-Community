# MCP Project Overview

## What is This Project?

This project demonstrates how to expose custom tools as MCP servers to Large Language Models using the **Model Context Protocol (MCP)**. It enables the LLMs (such as Azure OpenAI GPT models) to call backend Python functions, including Azure AI Search and a basic calculator, via a standardized protocol. The project includes both server and client/host scripts, and is designed for easy deployment to Azure using Azure Container Registry and Azure Container Apps.

---

## Key Features

- **MCP Server**: Exposes Python functions as tools for LLMs, including a calculator and Azure AI Search integration.
- **SSE & HTTP Support**: Real-time communication using Server-Sent Events (SSE), compatible with OpenAI SDK.
- **Azure AI Search Integration**: Enables LLMs to retrieve and reason over enterprise data indexed in Azure AI Search.
- **Test Scripts**: Includes OpenAI SDK-based test clients to simulate LLM tool-calling workflows.

---

## Project Structure

- [`server.py`](server.py): FastMCP server exposing tools (calculator, AI Search) using streamable-http transport.
- [`server_sse.py`](server_sse.py): MCP server exposing tools (calculator, AI Search) using Starlette and SSE ***for OpenAI SDK compatibility***.
- [`test_mcp_server/openai_sdk.py`](test_mcp_server/openai_sdk.py): Test script that simulates an LLM calling MCP tools via OpenAI SDK.
- [`test_mcp_server/semantic_kernel_agent.py`](test_mcp_server/semantic_kernel_agent.py): Example of using Semantic Kernel with MCP plugins.
- [`requirements.txt`](requirements.txt): Python dependencies for the MCP server and test scripts.
- [`Dockerfile`](Dockerfile): Container definition for deployment.
- [`docker_env/.env_example`](docker_env/.env_example): Example environment variables for local or container deployment.

---

## How It Works

1. **MCP Server** exposes tools (e.g., `add`, `ai_search`) via HTTP/SSE endpoints.
2. **LLM (Azure OpenAI)** is prompted with a user query and a list of available tools.
3. **LLM decides** which tool(s) to call, and the client script (e.g., [`openai_sdk.py`](test_mcp_server/openai_sdk.py)) orchestrates the tool invocation.
4. **Tool results** are returned to the LLM, which then generates a final response for the user.

---

## How to Deploy to Azure

### 1. Build and Push Docker Image to Azure Container Registry (ACR)

```sh
# Log in to Azure
az login

# Set variables
ACR_NAME=<your-acr-name>
IMAGE_NAME=mcp-server
TAG=latest

# Build and push image to ACR
az acr build --registry $ACR_NAME --image $IMAGE_NAME:$TAG .
```

### 2. Deploy to Azure Container Apps

```sh
# Update the container app to use the new image
az containerapp update \
  --name <your-container-app-name> \
  --resource-group <your-resource-group> \
  --image $ACR_NAME.azurecr.io/$IMAGE_NAME:$TAG
```

- Make sure to set environment variables in the Container App settings (see `.env_example` for required keys).

---

## Environment Variables

Copy and fill in [`ex_env_file`](ex_env_file) as `.env` for local development or set these as application settings in Azure.

Key variables include:
- `AZURE_SEARCH_ENDPOINT`
- `AZURE_SEARCH_KEY`
- `AZURE_SEARCH_INDEX`
- `AZURE_SEARCH_VECTOR_FIELD_NAME`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_KEY`
- `AZURE_OPENAI_VERSION`
- `AZURE_EMBEDDINGS_DEPLOYMENT`

---

## Running Locally

You can use [uv](https://github.com/astral-sh/uv) as a fast, reliable drop-in replacement for `pip` and `python` commands below.

1. **Install dependencies** (recommended: use `uv`):

   ```sh
   uv pip install -r requirements.txt
   # or, with pip:
   pip install -r requirements.txt
   ```

2. **Set up your `.env` file** (copy from `.env_example` and fill in values).

3. **Start the server** (for example, with SSE support):

   ```sh
   uv python server_sse.py
   # or, with python:
   python server_sse.py
   ```

4. **Run test scripts** (in another terminal):

   ```sh
   uv python test_mcp_server/openai_sdk.py
   # or, with python:
   python test_mcp_server/openai_sdk.py
   ```

---

## References

- [MCP Protocol GitHub](https://github.com/microsoft/mcp-for-beginners)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)

---

**This project enables secure, scalable, and extensible tool-calling for LLMs in enterprise environments using Azure.**