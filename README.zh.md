
1. Introduction:
Data-domain large model framework. Its purpose is to build the infrastructure for the large model domain by developing a variety of technical capabilities, including multi-model management, Text2SQL performance optimization, RAG framework and optimization. These capabilities aim to simplify and facilitate the construction of large model applications around databases.

2. Installation:

3. Usage:

4. Directory Structure:

- /assets: Script for creating db schema for storing all metadata of the application 
- /proj : Contains the code for python server
  - /_private : Contains code to read all the configuration from .env file
  - /_agents : Contains code for type of display command whether the command is to display chart , table or text and all code related to displaying these.
  - /app : 
    - /initialization:  initialization of embedding model
    - /knowledge : for creating and storing knowledge space along with chunking and splitting
    - /openapi : APIs
    - /scene : Various type of chat interactions with db , excel , dashboard , knowledge etc
    - /static : Published UI code of next.js to be served as static content
    - /appserver.py : starting point of app server
  - /config : all the model configuration
  - /core :
  - /datasources : for connection with various databases and related methods to read metadata (table , colum and field info)
  - /model :
  - /serve : Middle service layer
  - /rag : contains all the code for generating embeddings , extracting and summarizing , retrieving based similarity search etc , text splitter
  - /storage : vector store connectors and loading and retrieving queries from vector store , metadata storage in sqlite storing all chat history , humand feedback , configurations etc in 
- /web : UI code in next.js