'''
MCP (Model Context Protocol) Server with Streamable-HTTP Support

This server provides tools to Large Language Models through the MCP protocol.
It uses streamable-http for real-time communication, which is NOT compatible with OpenAI's SDK to date.

Features:
- Simple calculator tool (add function)
- Azure AI Search integration for retrieving documents
- SSE transport for real-time communication
- Starlette web framework for HTTP handling

**COMMAND TO BUILD AND PUSH DOCKER IMAGE TO ACR:
az acr build --registry <your-acr-name> --image <your-image-name>:<tag> .

**COMMAND TO UPDATE THE CONTAINER APP TO USE THE NEW IMAGE:
az containerapp update --name <your-container-app-name> --resource-group <your-resource-group> --image <your-acr-name>.azurecr.io/<your-image-name>:<tag>

'''

from mcp.server.fastmcp import FastMCP
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential  
from azure.search.documents import SearchClient 
from azure.search.documents.models import VectorizedQuery

load_dotenv()

# Create an MCP server
mcp = FastMCP(
    name="Calculator",
    host="0.0.0.0",  # only used for SSE transport (localhost)
    port=8080,  # only used for SSE transport (set this to any port)
)


# Add a simple calculator tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b

# Add a tool to retrieve documents from an Azure AI Search index
@mcp.tool()
def ai_search(query: str) -> str:
    """
    Retrieve documents from an Azure AI Search index.

    Args:
        query (str): The search query string.

    Returns:
        str: A list of documents matching the query in a string.
    """
    # Set env variables
    search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    search_key = os.getenv("AZURE_SEARCH_KEY")
    search_index = os.getenv("AZURE_SEARCH_INDEX")
    search_vector_field_name = os.getenv("AZURE_SEARCH_VECTOR_FIELD_NAME")    
    search_content_field_name = os.getenv("AZURE_SEARCH_CONTENT_FIELD_NAME")    
    azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')  
    azure_openai_api_key = os.getenv('AZURE_OPENAI_KEY')  
    azure_openai_api_version = os.getenv('AZURE_OPENAI_VERSION')  
    azure_ada_deployment = os.getenv('AZURE_EMBEDDINGS_DEPLOYMENT')  
    # Initialize the AzureOpenAI client
    client = AzureOpenAI(
        api_key=azure_openai_api_key,
        api_version=azure_openai_api_version,
        azure_endpoint=azure_endpoint
    )
    # Get the vector representation of the user query using the Ada model
    embedding_response = client.embeddings.create(
        input=query,
        model=azure_ada_deployment
    )
    query_vector = embedding_response.data[0].embedding
    # Create a SearchClient  
    search_client = SearchClient(endpoint=search_endpoint,  
                                index_name=search_index,  
                                credential=AzureKeyCredential(search_key))  
    vector_query = VectorizedQuery(vector=query_vector, k_nearest_neighbors=5, fields=search_vector_field_name, exhaustive=True)
    # Query the index  
    results = search_client.search(
        search_text=None,
        vector_queries= [vector_query],
        top=5,
        select="*",
        semantic_query=query,
        query_type="semantic",
        semantic_configuration_name="itsupportindex-semantic-configuration",  # Ensure this matches your index configuration
        # query_language="en-us",
        query_answer="extractive|count-5",
        query_caption="extractive|highlight-false",
        )  
        # Print the results 
    i = 0 
    final_output = ""
    for result in results:
        reranker_score = result.get('@search.reranker_score', 0)  # Default to 0 if not present
        if reranker_score >= 1.5:  # Filter by reranker score
            i += 1 
            content = result.get(search_content_field_name, '')
            score = result.get('@search.score', 'N/A')
            final_output += (
                f"Source {i}\n"
                f"Content: {content}\n"
                f"@search.score: {score}\n"
                f"@search.reranker_score: {reranker_score}\n"
                f"{'-'*50}\n\n"
            )
    return final_output

    
    


# Run the server
if __name__ == "__main__":
    transport = "streamable-http"  
    if transport == "stdio":
        print("Running server with stdio transport")
        mcp.run(transport="stdio")
    elif transport == "streamable-http":
        print("Running server with SSE transport")
        mcp.run(transport="streamable-http")
    else:
        raise ValueError(f"Unknown transport: {transport}")