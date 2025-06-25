'''
MCP (Model Context Protocol) Server with SSE (Server-Sent Events) Support

This server provides tools to Large Language Models through the MCP protocol.
It uses SSE for real-time communication, which is compatible with OpenAI's SDK.

Features:
- Simple calculator tool (add function)
- Azure AI Search integration for retrieving documents
- SSE transport for real-time communication
- Starlette web framework for HTTP handling

Technical Note: SSE is deprecated in favor of streamable-http, but OpenAI SDK 
hasn't updated yet, so we use Starlette to handle SSE requests.

**COMMAND TO BUILD AND PUSH DOCKER IMAGE TO ACR:
az acr build --registry <your-acr-name> --image <your-image-name>:<tag> .

**COMMAND TO UPDATE THE CONTAINER APP TO USE THE NEW IMAGE:
az containerapp update --name <your-container-app-name> --resource-group <your-resource-group> --image <your-acr-name>.azurecr.io/<your-image-name>:<tag>

'''

# MCP (Model Context Protocol) imports
from mcp.server.fastmcp import FastMCP         
from mcp.server import Server                   
from mcp.server.sse import SseServerTransport   

# Web framework imports
from starlette.applications import Starlette    
from starlette.routing import Mount, Route      
from starlette.requests import Request          
import uvicorn                                 

# Standard library and utility imports
import argparse                                 
from typing import Any                         
import httpx, os, random                        
import os                                       
from dotenv import load_dotenv                  

# Azure services imports
from openai import AzureOpenAI                           
from azure.core.credentials import AzureKeyCredential   
from azure.search.documents import SearchClient         
from azure.search.documents.models import VectorizedQuery  

# Load environment variables from .env file
load_dotenv()

# Create the main MCP server instance
# FastMCP provides a simplified way to create MCP servers with decorators
mcp = FastMCP(
    name="InfoHub",  # This name identifies our server to MCP clients
)


# Tool 1: Simple Calculator
# The @mcp.tool() decorator registers this function as an available tool for LLMs
@mcp.tool()
def add(a: int, b: int) -> int:
    """
    Add two numbers together.
    
    This is a simple example tool that demonstrates how to expose functions
    to Large Language Models through the MCP protocol.
    
    Args:
        a (int): First number to add
        b (int): Second number to add
        
    Returns:
        int: Sum of the two numbers
    """
    return a + b

# Tool 2: Azure AI Search Integration
# This tool allows LLMs to search through documents stored in Azure AI Search
@mcp.tool()
def ai_search(query: str) -> str:
    """
    Retrieve documents from an Azure AI Search index using hybrid search.
    
    This function performs a sophisticated search that combines:
    - Vector search (using embeddings for semantic similarity)
    - Semantic search (using Azure's built-in semantic capabilities)
    - Reranking (to improve result relevance)
    
    Args:
        query (str): The search query string from the user
        
    Returns:
        str: Formatted search results with content, scores, and metadata
    """
    # Step 1: Get environment variables for Azure services
    search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")        
    search_key = os.getenv("AZURE_SEARCH_KEY")                  
    search_index = os.getenv("AZURE_SEARCH_INDEX") 
    search_vector_field_name = os.getenv("AZURE_SEARCH_VECTOR_FIELD_NAME")    
    search_content_field_name = os.getenv("AZURE_SEARCH_CONTENT_FIELD_NAME")         
    azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')         
    azure_openai_api_key = os.getenv('AZURE_OPENAI_KEY')        
    azure_openai_api_version = os.getenv('AZURE_OPENAI_VERSION') 
    azure_ada_deployment = os.getenv('AZURE_EMBEDDINGS_DEPLOYMENT') 
    
    # Step 2: Initialize Azure OpenAI client for generating embeddings
    client = AzureOpenAI(
        api_key=azure_openai_api_key,
        api_version=azure_openai_api_version,
        azure_endpoint=azure_endpoint
    )
    
    # Step 3: Convert the user's query into a vector representation
    embedding_response = client.embeddings.create(
        input=query,
        model=azure_ada_deployment
    )
    query_vector = embedding_response.data[0].embedding
    
    # Step 4: Initialize Azure AI Search client
    search_client = SearchClient(
        endpoint=search_endpoint,  
        index_name=search_index,  
        credential=AzureKeyCredential(search_key)
    )
    
    # Step 5: Create a vector query for similarity search
    vector_query = VectorizedQuery(
        vector=query_vector, 
        k_nearest_neighbors=5,        
        fields=search_vector_field_name,         
        exhaustive=True               
    )
    
    # Step 6: Execute the hybrid search
    results = search_client.search(
        search_text=None,                    
        vector_queries=[vector_query],      
        top=5,                              
        select="*",                         
        semantic_query=query,               
        query_type="semantic",              
        semantic_configuration_name="itsupportindex-semantic-configuration", 
        query_answer="extractive|count-5",  
        query_caption="extractive|highlight-false",  
    )
    
    # Step 7: Process and format the search results
    i = 0 
    final_output = ""
    for result in results:
        # Get the reranker score (measures relevance quality)
        reranker_score = result.get('@search.reranker_score', 0)
        
        # Only include results with good reranker scores (>=1.5)
        if reranker_score >= 1.5:
            i += 1 
            content = result.get(search_content_field_name, '')           
            score = result.get('@search.score', 'N/A') 
            
            # Format the result in a readable way
            final_output += (
                f"Source {i}\n"
                f"Content: {content}\n"
                f"@search.score: {score}\n"
                f"@search.reranker_score: {reranker_score}\n"
                f"{'-'*50}\n\n"
            )
    
    return final_output 


# Web Application Setup
# This section creates a web application that can handle MCP connections over HTTP

def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    """
    Create a Starlette web application that serves the MCP server over SSE.
    
    This function sets up the web infrastructure needed to expose our MCP tools
    to clients (like LLMs) over HTTP using Server-Sent Events (SSE).
    
    Args:
        mcp_server: The MCP server instance containing our tools
        debug: Whether to enable debug mode (more verbose logging)
        
    Returns:
        Starlette: A configured web application ready to serve MCP requests
    """
    # Create SSE transport layer
    # SSE allows real-time bidirectional communication between client and server
    # The "/messages/" path is where MCP protocol messages will be sent
    sse = SseServerTransport("/messages/")

    class SSEEndpoint:
        """
        ASGI-compatible endpoint for handling SSE connections.
        
        This class implements the ASGI callable protocol, which means it can
        be used directly as a route endpoint in Starlette. It handles the
        SSE connection setup and delegates MCP communication to the server.
        """
        
        def __init__(self, mcp_server: Server, sse_transport: SseServerTransport):
            """
            Initialize the SSE endpoint.
            
            Args:
                mcp_server: The MCP server that will handle tool requests
                sse_transport: The SSE transport layer for communication
            """
            self.mcp_server = mcp_server
            self.sse_transport = sse_transport
            
        async def __call__(self, scope, receive, send):
            """
            Handle incoming SSE connections (ASGI callable protocol).
            
            This method is called when a client connects to the /sse endpoint.
            It establishes the SSE connection and runs the MCP server to handle
            tool requests from the client.
            
            Args:
                scope: ASGI scope containing request information
                receive: ASGI receive function for getting messages
                send: ASGI send function for sending responses
            """
            print(f"{'-'*50}\nhandling sse\n{'-'*50}")  # Log when a new SSE connection is established
            
            # Establish SSE connection and run MCP server
            # The context manager handles connection setup/teardown automatically
            async with self.sse_transport.connect_sse(scope, receive, send) as (read_stream, write_stream):
                # Run the MCP server with the established streams
                # This is where the actual MCP protocol communication happens
                await self.mcp_server.run(
                    read_stream,                                    # Stream for receiving messages
                    write_stream,                                   # Stream for sending responses
                    self.mcp_server.create_initialization_options(), # Server configuration
                )

    # Create and configure the Starlette web application
    return Starlette(
        debug=debug,  # Enable debug mode if requested
        routes=[
            # Route for SSE connections - this is where clients connect for real-time communication
            Route("/sse", endpoint=SSEEndpoint(mcp_server, sse)),
            
            # Route for MCP message handling - this handles the actual protocol messages
            # Mount means all URLs under /messages/ will be handled by this app
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )


# Server Startup Section
# This section handles starting the server when the script is run directly

if __name__ == "__main__":
    # Get the underlying MCP server instance from FastMCP
    # FastMCP is a wrapper that simplifies MCP server creation
    mcp_server = mcp._mcp_server  

    # Create the web application that will serve our MCP server
    # This wraps our MCP server in a web interface accessible over HTTP
    starlette_app = create_starlette_app(mcp_server, debug=True)

    # Set up command-line argument parsing
    # This allows users to customize host and port when running the server
    parser = argparse.ArgumentParser(description='Run MCP SSE-based server')
    parser.add_argument('--host', default='0.0.0.0', 
                       help='Host to bind to (0.0.0.0 means accept connections from any IP)')
    parser.add_argument('--port', type=int, default=8080, 
                       help='Port to listen on (default: 8080)')
    args = parser.parse_args()

    # Start the web server
    # Uvicorn is an ASGI server that can run our Starlette application
    print(f"Starting MCP server on {args.host}:{args.port}")
    print(f"Available tools: add, ai_search")
    print(f"SSE endpoint: http://{args.host}:{args.port}/sse")
    print(f"Messages endpoint: http://{args.host}:{args.port}/messages/")
    
    uvicorn.run(starlette_app, host=args.host, port=args.port)