# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC # 1/ Ingesting and preparing PDF for LLM and Pinecone Vector Database
# MAGIC
# MAGIC ## In this example, we will focus on ingesting pdf documents as source for our retrieval process. 
# MAGIC
# MAGIC <img src="https://github.com/prasadkona/databricks_demos/blob/main/images/llm-rag-full-pinecone-1.png?raw=true" style="float: right; width: 600px; margin-left: 10px">
# MAGIC
# MAGIC
# MAGIC For this example, we will add Databricks ebook PDFs from [Databricks resources page](https://www.databricks.com/resources) to our knowledge database.
# MAGIC
# MAGIC **Note: This demo is an advanced content, we strongly recommand going over the simple version first to learn the basics.**
# MAGIC
# MAGIC Here are all the detailed steps:
# MAGIC
# MAGIC - Use autoloader to load the binary PDF as our first table. 
# MAGIC - Use the `unstructured` library  to parse the text content of the PDFs.
# MAGIC - Use `llama_index` or `Langchain` to split the texts into chuncks.
# MAGIC - Compute embeddings for the chunks
# MAGIC - Save our text chunks + embeddings in a Delta Lake table
# MAGIC - Write to Pinecone vector database.
# MAGIC
# MAGIC
# MAGIC Lakehouse AI not only provides state of the art solutions to accelerate your AI and LLM projects, but also to accelerate data ingestion and preparation at scale, including unstructured data like pdfs.
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection or disable tracker during installation. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=1444828305810485&notebook=advanced/01-PDF-Advanced-Data-Preparation&demo_name=chatbot-rag-llm&event=VIEW">

# COMMAND ----------

# DBTITLE 1,Install required external libraries 
# MAGIC %pip install transformers==4.30.2 "unstructured[pdf,docx]==0.10.30" langchain==0.1.5 llama-index==0.9.3 pydantic==1.10.9 mlflow==2.10.1 pinecone-client==4.1.0
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../_resources/00-init-advanced $reset_all_data=false

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Ingesting Databricks ebook PDFs and extracting their pages
# MAGIC
# MAGIC <img src="https://github.com/prasadkona/databricks_demos/blob/main/images/llm-rag-full-pinecone-2.png?raw=true" style="float: right" width="500px">
# MAGIC
# MAGIC First, let's ingest our PDFs as a Delta Lake table with path urls and content in binary format. 
# MAGIC
# MAGIC We'll use [Databricks Autoloader](https://docs.databricks.com/en/ingestion/auto-loader/index.html) to incrementally ingest new files, making it easy to incrementally consume billions of files from the data lake in various data formats. Autoloader can easily ingests our unstructured PDF data in binary format.
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE VOLUME IF NOT EXISTS volume_databricks_documentation;

# COMMAND ----------

# DBTITLE 1,Our pdf or docx files are available in our Volume (or DBFS)
# List our raw PDF docs
volume_folder =  f"/Volumes/{catalog}/{db}/volume_databricks_documentation"
#Let's upload some pdf to our volume as example
upload_pdfs_to_volume(volume_folder+"/databricks-pdf")

display(dbutils.fs.ls(volume_folder+"/databricks-pdf"))

# COMMAND ----------

# DBTITLE 1,Ingesting PDF files as binary format using Databricks Autoloader
df = (spark.readStream
        .format('cloudFiles')
        .option('cloudFiles.format', 'BINARYFILE')
        .load('dbfs:'+volume_folder+"/databricks-pdf"))

# Write the data as a Delta table
(df.writeStream
  .trigger(availableNow=True)
  .option("checkpointLocation", f'dbfs:{volume_folder}/checkpoints/raw_docs')
  .table('pdf_raw').awaitTermination())

# COMMAND ----------

# MAGIC %sql SELECT * FROM pdf_raw LIMIT 2

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <img src="https://github.com/prasadkona/databricks_demos/blob/main/images/llm-rag-full-pinecone-3.png?raw=true" style="float: right" width="600px">
# MAGIC
# MAGIC ## Extracting our PDF content as text chunk
# MAGIC
# MAGIC We need to convert the pdf documents bytes as text, and extract chunks from their content.
# MAGIC
# MAGIC This part can be tricky as pdf are hard to work with and can be saved as images, for which we'll need an OCR to extract the text.
# MAGIC
# MAGIC Using the `Unstructured` library within a Spark UDF makes it easy to extract text. 
# MAGIC
# MAGIC *Note: Your cluster will need a few extra libraries that you would typically install with a cluster init script.*
# MAGIC
# MAGIC <br style="clear: both">
# MAGIC
# MAGIC ### Splitting our big documentation page in smaller chunks
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/chunk-window-size.png?raw=true" style="float: right" width="700px">
# MAGIC
# MAGIC In this demo, some PDF can be really big, with a lot of text.
# MAGIC
# MAGIC We'll extract the content and then use llama_index `SentenceSplitter`, and ensure that each chunk isn't bigger than 500 tokens. 
# MAGIC
# MAGIC
# MAGIC The chunk size and chunk overlap depend on the use case and the PDF files. 
# MAGIC
# MAGIC Remember that your prompt+answer should stay below your model max window size (4096 for llama2). 
# MAGIC
# MAGIC For more details, review the previous [../01-Data-Preparation](01-Data-Preparation) notebook. 
# MAGIC
# MAGIC <br/>
# MAGIC <br style="clear: both">
# MAGIC <div style="background-color: #def2ff; padding: 15px;  border-radius: 30px; ">
# MAGIC   <strong>Information</strong><br/>
# MAGIC   Remember that the following steps are specific to your dataset. This is a critical part to building a successful RAG assistant.
# MAGIC   <br/> Always take time to review the chunks created and ensure they make sense, containing relevant informations.
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,To extract our PDF,  we'll need to setup libraries in our nodes
#For production use-case, install the libraries at your cluster level with an init script instead. 
install_ocr_on_nodes()

# COMMAND ----------

# MAGIC %md 
# MAGIC Let's start by extracting text from our PDF.

# COMMAND ----------

# DBTITLE 1,Transform pdf as text
from unstructured.partition.auto import partition
import re
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

def extract_doc_text(x : bytes) -> str:
  # Read files and extract the values with unstructured
  sections = partition(file=io.BytesIO(x))
  def clean_section(txt):
    txt = re.sub(r'\n', '', txt)
    return re.sub(r' ?\.', '.', txt)
  # Default split is by section of document, concatenate them all together because we want to split by sentence instead.
  return "\n".join([clean_section(s.text) for s in sections]) 

# COMMAND ----------

#Let's try our text extraction function with a single pdf file
import io
import re
with requests.get('https://github.com/databricks-demos/dbdemos-dataset/blob/main/llm/databricks-pdf-documentation/Databricks-Customer-360-ebook-Final.pdf?raw=true') as pdf:
  doc = extract_doc_text(pdf.content)  
  print(doc)

# COMMAND ----------

# MAGIC %md
# MAGIC This looks great. We'll now wrap it with a text_splitter to avoid having too big pages, and create a Pandas UDF function to easily scale that across multiple nodes.
# MAGIC
# MAGIC *Note that our pdf text isn't clean. To make it nicer, we could imagine a few extra LLM-based pre-processing steps, asking to remove unrelevant content like the list of chapters to only keep the meat of the text.*

# COMMAND ----------

from llama_index.langchain_helpers.text_splitter import SentenceSplitter
from llama_index import Document, set_global_tokenizer
from transformers import AutoTokenizer

# Reduce the arrow batch size as our PDF can be big in memory
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", 10)

@pandas_udf("array<string>")
def read_as_chunk(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    #set llama2 as tokenizer
    set_global_tokenizer(
      AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    )
    #Sentence splitter from llama_index to split on sentences
    splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)
    def extract_and_split(b):
      txt = extract_doc_text(b)
      nodes = splitter.get_nodes_from_documents([Document(text=txt)])
      return [n.text for n in nodes]

    for x in batch_iter:
        yield x.apply(extract_and_split)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## What's required for Pinecone Vector Database
# MAGIC
# MAGIC
# MAGIC In this demo, we will show you how to use pinecone as your vector database
# MAGIC
# MAGIC We will first compute the embeddings of our chunks and save them as a Delta Lake table field as `array&ltfloat&gt`

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Introducing Databricks BGE Embeddings Foundation Model endpoints
# MAGIC
# MAGIC <img src="https://github.com/prasadkona/databricks_demos/blob/main/images/llm-rag-full-pinecone-5.png?raw=true" style="float: right; width: 600px; margin-left: 10px">
# MAGIC
# MAGIC Foundation Models are provided by Databricks, and can be used out-of-the-box.
# MAGIC
# MAGIC Databricks supports several endpoint types to compute embeddings or evaluate a model:
# MAGIC - A **foundation model endpoint**, provided by databricks (ex: llama2-70B, MPT...)
# MAGIC - An **external endpoint**, acting as a gateway to an external model (ex: Azure OpenAI)
# MAGIC - A **custom**, fined-tuned model hosted on Databricks model service
# MAGIC
# MAGIC Open the [Model Serving Endpoint page](/ml/endpoints) to explore and try the foundation models.
# MAGIC
# MAGIC For this demo, we will use the foundation model `BGE` (embeddings) and `llama2-70B` (chat). <br/><br/>
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/databricks-foundation-models.png?raw=true" width="600px" >

# COMMAND ----------

# DBTITLE 1,Using Databricks Foundation model BGE as embedding endpoint
from mlflow.deployments import get_deploy_client

# bge-large-en Foundation models are available using the /serving-endpoints/databricks-bge-large-en/invocations api. 
deploy_client = get_deploy_client("databricks")

## NOTE: if you change your embedding model here, make sure you change it in the query step too
embeddings = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": ["What is Apache Spark?"]})
print(embeddings)

# COMMAND ----------

# DBTITLE 1,Create the final databricks_pdf_documentation table containing chunks and embeddings
# MAGIC %sql
# MAGIC --Note that we need to enable Change Data Feed on the table to create the index
# MAGIC CREATE TABLE IF NOT EXISTS databricks_pdf_documentation (
# MAGIC   id BIGINT GENERATED BY DEFAULT AS IDENTITY,
# MAGIC   url STRING,
# MAGIC   content STRING,
# MAGIC   metadata STRING,
# MAGIC   embedding ARRAY <FLOAT>
# MAGIC ) TBLPROPERTIES (delta.enableChangeDataFeed = true); 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Computing the chunk embeddings and saving them to our Delta Table
# MAGIC
# MAGIC The last step is to now compute an embedding for all our documentation chunks. Let's create an udf to compute the embeddings using the foundation model endpoint.
# MAGIC
# MAGIC *Note that this part would typically be setup as a production-grade job, running as soon as a new documentation page is updated. <br/> This could be setup as a Delta Live Table pipeline to incrementally consume updates.*

# COMMAND ----------

import pandas as pd
import numpy as np
from pyspark.sql.functions import pandas_udf,PandasUDFType, udf

@pandas_udf("array<float>")
def get_embedding(contents: pd.Series) -> pd.Series:
    import mlflow.deployments
    deploy_client = mlflow.deployments.get_deploy_client("databricks")
    def get_embeddings(batch):
        #Note: this will gracefully fail if an exception is thrown during embedding creation (add try/except if needed) 
        response = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": batch})
        return [e['embedding'] for e in response.data]

    # Splitting the contents into batches of 150 items each, since the embedding model takes at most 150 inputs per request.
    max_batch_size = 150
    batches = [contents.iloc[i:i + max_batch_size] for i in range(0, len(contents), max_batch_size)]

    # Process each batch and collect the results
    all_embeddings = []
    for batch in batches:
        all_embeddings += get_embeddings(batch.tolist())

    return pd.Series(all_embeddings)

# COMMAND ----------

import json
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StringType
import pandas as pd

@pandas_udf(StringType())
def create_metadata_json_string(original_col: pd.Series) -> pd.Series:
    json_dict = {'original_doc': original_col}

    return pd.Series(json.dumps(json_dict))

#df = df.withColumn('json_col', create_metadata_json_string('input_col'))



# COMMAND ----------

# UDF for embedding
from pyspark.sql.types import *
def get_embedding_for_string(text):
    response = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": text})
    e = response.data
    return e[0]['embedding']

get_embedding_for_string_udf = udf(get_embedding_for_string, ArrayType(FloatType()))
print(get_embedding_for_string("What is a lakehouse ?"))

# COMMAND ----------

# Delete checkpoint for the pdf_raw table streaming query
dbutils.fs.rm(f'{folder}/checkpoints/pdf_chunks_{catalog}_{db}', True)

# Delete checkpoint for the databricks_documentation table streaming query
dbutils.fs.rm(f'{folder}/checkpoints/docs_chunks_{catalog}_{db}', True)

# COMMAND ----------

(spark.readStream.table('pdf_raw')
      .withColumn("content", F.explode(read_as_chunk("content")))
      .withColumn("embedding", get_embedding("content"))
      .withColumn("metadata", create_metadata_json_string("content") )
      #.selectExpr('path as url', 'content', 'embedding','metadata')
      .selectExpr('path as url', 'content', 'embedding')
  .writeStream
    .trigger(availableNow=True)
    .option("checkpointLocation", f'{folder}/checkpoints/pdf_chunks_{catalog}_{db}')
    .table('databricks_pdf_documentation').awaitTermination())

#Let's also add our documentation web page from the simple demo (make sure you run the simple demo for it to work)
if spark.catalog.tableExists(f'{catalog}.{db}.databricks_documentation'):
  (spark.readStream.table('databricks_documentation')
      .withColumn('embedding', get_embedding("content"))
      .withColumn("metadata", create_metadata_json_string("content") )
      #.select('url', 'content', 'embedding','metadata')
      .select('url', 'content', 'embedding')
  .writeStream
    .trigger(availableNow=True)
    .option("checkpointLocation", f'{folder}/checkpoints/docs_chunks_{catalog}_{db}')
    .table('databricks_pdf_documentation').awaitTermination())

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM databricks_pdf_documentation WHERE url like '%.pdf' limit 10

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType, IntegerType  
schema = StructType([  
    StructField("id",StringType(),True),  
    StructField("values",ArrayType(FloatType()),True),  
    StructField("namespace",StringType(),True),  
    StructField("metadata", StringType(), True),  
    StructField("sparse_values", StructType([  
        StructField("indices", ArrayType(IntegerType(), False), False),  
        StructField("values", ArrayType(FloatType(), False), False)  
    ]), True)  
])  
#embeddings_df = spark.createDataFrame(data=embeddings,schema=schema)  


# COMMAND ----------

from pyspark.sql.functions import col, lit, struct, to_json
from pyspark.sql.functions import encode

df = spark.table('databricks_pdf_documentation')\
            .withColumn("metadata", to_json(struct(col("content"), col("url"), col("id"))))\
            .withColumn("namespace", lit("dbdemo-namespace")) \
            .withColumn("values", col("embedding")) \
            .withColumn("sparse_values", lit(None)) \
            .select("id", "values", "namespace", "metadata", "sparse_values")

display(df.count())

# Print the valid JSON
display(df.limit(2))

# COMMAND ----------

# If you dont know the embedding array size, use the below to determine the embedding array size.
# The embedding array size varies based on the model used for converting a string to an embedding array
# Note: Login to pinecone, Set the pinecone vector index to have the size of the embedding array 

from pyspark.sql.functions import size

df2 = df.withColumn('array_col_len', size('values'))
display(df2.limit(1))



# COMMAND ----------

# DBTITLE 1,Initialize Pinecone client configs
# Initialize pinecone variables

api_key = dbutils.secrets.get("prasad_kona", "PINECONE_API_KEY")
project_name = "Starter"
index_name = "dbdemo-index"



# COMMAND ----------

(  
    df.write  
    .option("pinecone.apiKey", api_key) 
    .option("pinecone.indexName", index_name)  
    .format("io.pinecone.spark.pinecone.Pinecone")  
    .mode("append")  
    .save()  
)  


# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### Our dataset is now ready! and is available for query via Pinecone
# MAGIC
# MAGIC Our dataset is now ready. We chunked the documentation page in small section, computed the embeddings and saved it as a Delta Lake table and ingested it into the Pinecone vector database

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Searching for similar content
# MAGIC
# MAGIC Let's give it a try and search for similar content. Lets get the top n results 
# MAGIC
# MAGIC

# COMMAND ----------


# connect to pinecone index
from pinecone import Pinecone

pc = Pinecone(api_key=api_key)
index = pc.Index(index_name)


# COMMAND ----------

question = "How can I track billing usage on my workspaces?"

# create the query embedding
xq = get_embedding_for_string(question)

# query pinecone the top 5 most similar results
query_response = index.query(
    namespace='dbdemo-namespace',
    top_k=5,
    include_values=True,
    include_metadata=True,
    vector=xq
)

#print(query_response)

query_response_docs = []
for match in query_response['matches']:
    query_response_docs.append([match['metadata']['url'],match['metadata']['content'],match['score']])

print(query_response_docs)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Next step: Deploy our chatbot model with RAG
# MAGIC
# MAGIC We've seen how Databricks Lakehouse AI makes it easy to ingest and prepare your documents, and write to Pinecone vector database.
# MAGIC
# MAGIC This simplifies and accelerates your data projects so that you can focus on the next step: creating your realtime chatbot endpoint with well-crafted prompt augmentation.
# MAGIC
# MAGIC Open the [02-Advanced-Chatbot-Chain]($./02-Advanced-Chatbot-Chain) notebook to create and deploy a chatbot endpoint.
