# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # 2/ Advanced chatbot with message history and filter using Langchain
# MAGIC
# MAGIC <img src="https://github.com/prasadkona/databricks_demos/blob/main/images/llm-rag-full-pinecone-0i.png?raw=true" style="float: right; margin-left: 10px"  width="900px;">
# MAGIC
# MAGIC Data is now available on the Pinecone vector database!
# MAGIC
# MAGIC Let's now create a more advanced langchain model to perform RAG.
# MAGIC
# MAGIC We will improve our langchain model with the following:
# MAGIC
# MAGIC - Build a complete chain supporting a chat history, using llama 2 input style
# MAGIC - Add a filter to only answer Databricks-related questions
# MAGIC - Compute the embeddings with Databricks BGE models within our chain to query the Pinecone vector database
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection or disable tracker during installation. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=1444828305810485&notebook=02-Deploy-RAG-Chatbot-Model&demo_name=chatbot-rag-llm&event=VIEW">
# MAGIC

# COMMAND ----------

# MAGIC %pip install -U mlflow==2.15.1 pinecone-client==5.0.1 langchain-pinecone==0.1.3 langchain==0.2.0 databricks-sdk==0.30.0 langchain-community==0.2.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import requests
import mlflow
import langchain

# url used to send the request to your model from the serverless endpoint
host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")

pinecone_index_name = "dbdemo-index"
pinecone_namespace = 'dbdemo-namespace'
pinecone_api_key = dbutils.secrets.get("pinecone_secrets_scope", "PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = dbutils.secrets.get("pinecone_secrets_scope", "PINECONE_API_KEY")
#os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get("pinecone_secrets_scope", "DATABRICKS_TOKEN")
#pinecone_api_key = os.environ["PINECONE_API_KEY"]

catalog = "prasad_kona_dev"
db = "rag_chatbot_prasad_kona"

# Set a debug flag
debug_flag = True


# COMMAND ----------

# MAGIC %md 
# MAGIC ## Register the chatbot model to Unity Catalog

# COMMAND ----------



# Specify the full path to the chain notebook
chain_notebook_file = "2.1 - Advanced-Chatbot-Chain - Using Pinecone"
chain_notebook_path = os.path.join(os.getcwd(), chain_notebook_file)

print(f"Chain notebook path: {chain_notebook_path}")

# COMMAND ----------

# DBTITLE 1,Provide the signature for the chain model
from mlflow.models import infer_signature
# Provide an example of the input schema that is used to set the MLflow model's signature

#print(f'Testing with relevant history and question...')
dialog = {
    "messages": [
        {"role": "user", "content": "What is Apache Spark?"}, 
        {"role": "assistant", "content": "Apache Spark is an open-source data processing engine that is widely used in big data analytics."}, 
        {"role": "user", "content": "Does it support streaming?"},
        {"role": "assistant", "content": "Yes."},
        {"role": "user", "content": "Tell me more about it's capabilities."},
    ]
}

dialog_output = {'result': "Delta Lake is an open-source storage layer that brings ACID transactions, scalable metadata handling, and unifies batch and streaming processing. It's built on top of Apache Parquet format, providing features like schema enforcement, data versioning (Time Travel), and scalable metadata handling. Delta Lake supports both batch and streaming workloads, making it a unified solution for big data processing. It also offers features like ACID transactions and schema evolution, which are crucial for maintaining data integrity and handling continuously changing data. Delta Lake is designed to handle petabyte-scale tables with billions of partitions and files, making it suitable for large-scale data processing.", 'sources': ['dbfs:/Volumes/prasad_kona_dev/rag_chatbot_prasad_kona/volume_databricks_documentation/databricks-pdf/building-reliable-data-lakes-at-scale-with-delta-lake.pdf', 'dbfs:/Volumes/prasad_kona_dev/rag_chatbot_prasad_kona/volume_databricks_documentation/databricks-pdf/building-reliable-data-lakes-at-scale-with-delta-lake.pdf', 'dbfs:/Volumes/prasad_kona_dev/rag_chatbot_prasad_kona/volume_databricks_documentation/databricks-pdf/The-Delta-Lake-Series-Lakehouse-012921.pdf', 'dbfs:/Volumes/prasad_kona_dev/rag_chatbot_prasad_kona/volume_databricks_documentation/databricks-pdf/big-book-of-data-engineering-2nd-edition-final.pdf']}

input_example = {
   "messages": [
       {
           "role": "user",
           "content": "How does billing work on Databricks?",
       }
   ]
}

output_example = {'result': "Databricks operates on a pay-as-you-go model, where you are billed based on the usage of cloud resources. The cost depends on the type and duration of cloud resources you use, such as compute instances and storage. You can monitor your usage and costs through the Databricks platform. For more specific billing details, I would recommend checking Databricks' official documentation or contacting their support.", 'sources': ['dbfs:/Volumes/prasad_kona_dev/rag_chatbot_prasad_kona/volume_databricks_documentation/databricks-pdf/EB-Ingesting-Data-FINAL.pdf', 'dbfs:/Volumes/prasad_kona_dev/rag_chatbot_prasad_kona/volume_databricks_documentation/databricks-pdf/big-book-of-data-and-ai-use-cases-for-the-public-sector.pdf', 'dbfs:/Volumes/prasad_kona_dev/rag_chatbot_prasad_kona/volume_databricks_documentation/databricks-pdf/big-book-of-data-and-ai-use-cases-for-the-public-sector.pdf', 'dbfs:/Volumes/prasad_kona_dev/rag_chatbot_prasad_kona/volume_databricks_documentation/databricks-pdf/Databricks-Customer-360-ebook-Final.pdf']}

signature = infer_signature(input_example, output_example)

# COMMAND ----------


mlflow.set_registry_uri("databricks-uc")
model_name = f"{catalog}.{db}.rag_with_pinecone_model"

with mlflow.start_run():
    signature = infer_signature(input_example, output_example)
    logged_chain_info = mlflow.langchain.log_model(
        lc_model=chain_notebook_path,
        artifact_path="chain",
        registered_model_name=model_name,
        input_example=input_example,
        signature=signature,
        example_no_conversion=True, # required to allow the schema to work
        extra_pip_requirements=[ 
          "mlflow==" + mlflow.__version__,
          "langchain==0.2.0" ,
          "pinecone-client==5.0.1",
          "langchain-pinecone==0.1.3",
          "langchain-community==0.2.0"
        ]
    )

# COMMAND ----------

logged_chain_info.model_uri

# COMMAND ----------

# MAGIC %md Let's try loading our model

# COMMAND ----------

logged_chain_info.model_uri

# COMMAND ----------

model = mlflow.langchain.load_model(logged_chain_info.model_uri)
model.invoke(dialog)

# COMMAND ----------

model.invoke(input_example)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Deploying our Chat Model as a Serverless Model Endpoint 
# MAGIC
# MAGIC Our model is saved in Unity Catalog. The last step is to deploy it as a Model Serving.
# MAGIC
# MAGIC We'll then be able to sending requests from our assistant frontend.

# COMMAND ----------

def get_latest_model_version(model_name):
    from mlflow import MlflowClient
    mlflow_client = MlflowClient()
    latest_version = 1
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

# COMMAND ----------

model = mlflow.langchain.load_model("models:/"+model_name+"/"+str(get_latest_model_version(model_name)))
model.invoke(dialog)

# COMMAND ----------


latest_model_version = get_latest_model_version(model_name)
print(latest_model_version)

# COMMAND ----------

# Create or update serving endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput, ServedModelInputWorkloadSize
import requests

serving_endpoint_name = "pinecone_rag_chain"
latest_model_version = get_latest_model_version(model_name)

databricks_api_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()


w = WorkspaceClient()
endpoint_config = EndpointCoreConfigInput(
    name=serving_endpoint_name,
    served_models=[
        ServedModelInput(
            model_name=model_name,
            model_version=latest_model_version,
            workload_size=ServedModelInputWorkloadSize.SMALL,
            scale_to_zero_enabled=True,
            environment_vars={
                "PINECONE_API_KEY": "{{secrets/prasad_kona/PINECONE_API_KEY}}",
                "DATABRICKS_TOKEN": "{{secrets/dbdemos/rag_sp_token}}",
            }
        )
    ]
)

existing_endpoint = next(
    (e for e in w.serving_endpoints.list() if e.name == serving_endpoint_name), None
)
serving_endpoint_url = f"{host}/ml/endpoints/{serving_endpoint_name}"
if existing_endpoint == None:
    print(f"Creating the endpoint {serving_endpoint_url}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.create_and_wait(name=serving_endpoint_name, config=endpoint_config)
else:
    print(f"Updating the endpoint {serving_endpoint_url} to version {latest_model_version}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.update_config_and_wait(served_models=endpoint_config.served_models, name=serving_endpoint_name)
    
displayHTML(f'Your Model Endpoint Serving is now available. Open the <a href="/ml/endpoints/{serving_endpoint_name}">Model Serving Endpoint page</a> for more details.')

# COMMAND ----------

# MAGIC %md
# MAGIC Our endpoint is now deployed! You can search endpoint name on the [Serving Endpoint UI](#/mlflow/endpoints) and visualize its performance!
# MAGIC
# MAGIC Let's run a REST query to try it in Python.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
serving_endpoint_name = "pinecone_rag_chain"
latest_model_version = get_latest_model_version(model_name)
print("latest_model_version="+str(latest_model_version))

# COMMAND ----------

# DBTITLE 1,Let's try to send a query to our chatbot
from databricks.sdk.service.serving import DataframeSplitInput

test_dialog = DataframeSplitInput(
    columns=["messages"],
    data=[
        
            {
                "messages": [
                    {"role": "user", "content": "What is Apache Spark?"},
                    {
                        "role": "assistant",
                        "content": "Apache Spark is an open-source data processing engine that is widely used in big data analytics.",
                    },
                    {"role": "user", "content": "Does it support streaming?"},
                ]
            }
        
    ],
)
answer = w.serving_endpoints.query(serving_endpoint_name, dataframe_split=test_dialog)
print(answer.predictions[0])

# COMMAND ----------

from databricks.sdk.service.serving import DataframeSplitInput

test_dialog = DataframeSplitInput(
    columns=["messages"],
    data=[
        
            {
                "messages": [
                    {"role": "user", "content": "How does billing work on Databricks?"},
                    
                ]
            }
        
    ],
)
answer = w.serving_endpoints.query(serving_endpoint_name, dataframe_split=test_dialog)
print(answer.predictions[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Congratulations! You have deployed your RAG application with Pinecone!
