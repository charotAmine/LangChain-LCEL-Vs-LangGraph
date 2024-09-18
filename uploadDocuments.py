from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SimpleField, SearchIndex,SearchableField
import os

# Azure Cognitive Search details
endpoint = ""  # Replace with your Azure search endpoint
api_key = ""    # Replace with your Azure search API key
index_name = ""  # Define your index name

# Initialize search client and index client
index_client = SearchIndexClient(endpoint=endpoint, credential=AzureKeyCredential(api_key))
search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(api_key))

# Define the search index schema
fields = [
    SimpleField(name="id", type="Edm.String", key=True),
    SearchableField(name="content", type="Edm.String")
]
index = SearchIndex(name=index_name, fields=fields)

# Create the index in Azure Cognitive Search
index_client.create_index(index)

# Load and upload documents from local .txt files
documents = []
file_directory = "./data"  # Directory where your .txt files are stored
for filename in os.listdir(file_directory):
    with open(os.path.join(file_directory, filename), "r") as file:
        content = file.read()
        documents.append({"id": filename.split(".")[0], "content": content})

# Upload documents to Azure Cognitive Search
search_client.upload_documents(documents)

print("Documents uploaded successfully.")
