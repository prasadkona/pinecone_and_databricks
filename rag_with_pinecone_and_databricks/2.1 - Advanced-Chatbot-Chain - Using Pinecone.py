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

# MAGIC %pip install mlflow==2.13.0 langchain==0.2.0 databricks-sdk==0.18.0 pydantic==2.5.2 pinecone-client==3.2.2 langchain-pinecone==0.1.1 lxml==4.9.3 cloudpickle==2.2.1 langchain-community==0.2.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import requests

# url used to send the request to your model from the serverless endpoint
#host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")

pinecone_index_name = "dbdemo-index"
pinecone_namespace = 'dbdemo-namespace'

catalog = "prasad_kona_dev"
db = "rag_chatbot_prasad_kona"

# Set a debug flag
# Set the debug flag to True to test this notebook
debug_flag = False

if debug_flag:
  ###### uncomment the next 3 lines to test this notebook
  ###### comment the next 3 lines to run the driver notebook for deploy to UC and model serving
  #pinecone_api_key = dbutils.secrets.get("prasad_kona", "PINECONE_API_KEY")
  #os.environ["PINECONE_API_KEY"] = dbutils.secrets.get("prasad_kona", "PINECONE_API_KEY")
  #os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get("prasad_kona", "DATABRICKS_TOKEN")
  pinecone_api_key = os.environ["PINECONE_API_KEY"]
else:
  pinecone_api_key = os.environ["PINECONE_API_KEY"]
  #pinecone_api_key = dbutils.secrets.get("prasad_kona", "PINECONE_API_KEY")
  #os.environ["PINECONE_API_KEY"] = dbutils.secrets.get("prasad_kona", "PINECONE_API_KEY")
  #os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get("prasad_kona", "DATABRICKS_TOKEN")

# COMMAND ----------

# MAGIC %md ### Try out Pinecone search

# COMMAND ----------

if debug_flag:
  from pinecone import Pinecone
  from langchain_pinecone import PineconeVectorStore
  from langchain_community.embeddings import DatabricksEmbeddings
  from pprint import pprint

  embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")


  # connect to pinecone index
  pc = Pinecone(api_key=pinecone_api_key)
  index_pc = pc.Index(pinecone_index_name)
  vectorstore = PineconeVectorStore(  
      index=index_pc,
      namespace=pinecone_namespace,
      embedding=embedding_model, 
      text_key="content"  
  )



# COMMAND ----------

if debug_flag:
  query = "What is Apache Spark?"
  docs = vectorstore.similarity_search(  
      query,  # our search query  
      k=3  # return 3 most relevant docs  
  )  
  pprint(docs[0])

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Exploring Langchain capabilities
# MAGIC
# MAGIC Let's start with the basics and send a query to a Databricks Foundation Model using LangChain.

# COMMAND ----------

# DBTITLE 1,Spark Chat Model Prompt
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatDatabricks
from langchain.schema.output_parser import StrOutputParser

prompt = PromptTemplate(
  input_variables = ["question"],
  template = "You are an assistant. Give a short answer ot this question: {question}"
)
chat_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 500)

chain = (
  prompt
  | chat_model
  | StrOutputParser()
)
# test the chain
if debug_flag:
  print(chain.invoke({"question": "What is Spark?"}))

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Adding conversation history to the prompt 

# COMMAND ----------

prompt_with_history_str = """
Your are a Big Data chatbot. Please answer Big Data question only. If you don't know or not related to Big Data, don't answer.

Here is a history between you and a human: {chat_history}

Now, please answer this question: {question}
"""

prompt_with_history = PromptTemplate(
  input_variables = ["chat_history", "question"],
  template = prompt_with_history_str
)

# COMMAND ----------

# MAGIC %md When invoking our chain, we'll pass history as a list, specifying whether each message was sent by a user or the assistant. For example:
# MAGIC
# MAGIC ```
# MAGIC [
# MAGIC   {"role": "user", "content": "What is Apache Spark?"}, 
# MAGIC   {"role": "assistant", "content": "Apache Spark is an open-source data processing engine that is widely used in big data analytics."}, 
# MAGIC   {"role": "user", "content": "Does it support streaming?"}
# MAGIC ]
# MAGIC ```
# MAGIC
# MAGIC Let's create chain components to transform this input into the inputs passed to `prompt_with_history`.

# COMMAND ----------

# DBTITLE 1,Chat History Extractor Chain
from langchain.schema.runnable import RunnableLambda
from operator import itemgetter

#The question is the last entry of the history
def extract_question(input):
    return input[-1]["content"]

#The history is everything before the last question
def extract_history(input):
    return input[:-1]

chain_with_history = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
    }
    | prompt_with_history
    | chat_model
    | StrOutputParser()
)

if debug_flag:
    print(chain_with_history.invoke({
        "messages": [
            {"role": "user", "content": "What is Apache Spark?"}, 
            {"role": "assistant", "content": "Apache Spark is an open-source data processing engine that is widely used in big data analytics."}, 
            {"role": "user", "content": "Does it support streaming?"}
        ]
    }))

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Let's add a filter on top to only answer Databricks-related questions.
# MAGIC
# MAGIC We want our chatbot to be profesionnal and only answer questions related to Databricks. Let's create a small chain and add a first classification step. 
# MAGIC
# MAGIC *Note: this is a fairly naive implementation, another solution could be adding a small classification model based on the question embedding, providing faster classification*

# COMMAND ----------

# DBTITLE 1,Databricks Inquiry Classifier
chat_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 200)

is_question_about_databricks_str = """
You are classifying documents to know if this question is related with Databricks in AWS, Azure and GCP, Workspaces, Databricks account and cloud infrastructure setup, Data Science, Data Engineering, Big Data, Datawarehousing, SQL, Python and Scala or something from a very different field. Also answer no if the last part is inappropriate. 

Here are some examples:

Question: Knowing this followup history: What is Databricks?, classify this question: Do you have more details?
Expected Response: Yes

Question: Knowing this followup history: What is Databricks?, classify this question: Write me a song.
Expected Response: No

Only answer with "yes" or "no". 

Knowing this followup history: {chat_history}, classify this question: {question}
"""

is_question_about_databricks_prompt = PromptTemplate(
  input_variables= ["chat_history", "question"],
  template = is_question_about_databricks_str
)

is_about_databricks_chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
    }
    | is_question_about_databricks_prompt
    | chat_model
    | StrOutputParser()
)

#test the chain
if debug_flag:

    #Returns "Yes" as this is about Databricks: 
    print(is_about_databricks_chain.invoke({
        "messages": [
            {"role": "user", "content": "What is Apache Spark?"}, 
            {"role": "assistant", "content": "Apache Spark is an open-source data processing engine that is widely used in big data analytics."}, 
            {"role": "user", "content": "Does it support streaming?"}
        ]
    }))

# COMMAND ----------

if debug_flag:
    #Return "no" as this isn't about Databricks
    print(is_about_databricks_chain.invoke({
        "messages": [
            {"role": "user", "content": "What is the meaning of life?"}
        ]
    }))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Use LangChain to retrieve documents from the Pinecone vector database
# MAGIC
# MAGIC
# MAGIC Let's add our Langchain retriever. 
# MAGIC
# MAGIC It will be in charge of:
# MAGIC
# MAGIC * Creating the input question embeddings (with Databricks `bge-large-en`)
# MAGIC * Calling the Pinecone vector database to find similar documents to augment the prompt with
# MAGIC
# MAGIC Langchain wrapper makes it easy to do in one step, handling all the underlying logic and API call for you.

# COMMAND ----------

from langchain.globals import set_debug
set_debug(False)

#host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

from langchain_community.embeddings import DatabricksEmbeddings
from langchain.chains import RetrievalQA
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

import os

from langchain.globals import set_debug
from langchain.globals import set_verbose

set_debug(False)
set_verbose(False)


embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

def get_retriever(persist_dir: str = None):
    #os.environ["DATABRICKS_HOST"] = host
    # initialize pinecone and connect to pinecone index
    pc = Pinecone(api_key=pinecone_api_key)
    index_pc = pc.Index(pinecone_index_name)

    vectorstore = PineconeVectorStore(  
        index=index_pc,
        namespace=pinecone_namespace,
        embedding=embedding_model, 
        text_key="content"  
    )

    return vectorstore.as_retriever()

retriever = get_retriever()

retrieve_document_chain = (
    itemgetter("messages") 
    | RunnableLambda(extract_question)
    | retriever
)

#test the retriever chain
if debug_flag:
    print(retrieve_document_chain.invoke({"messages": [{"role": "user", "content": "What is Apache Spark?"}]}))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Improve document search using LLM to generate a better sentence for the vector store, based on the chat history
# MAGIC
# MAGIC We need to retrieve documents related the the last question but also the history.
# MAGIC
# MAGIC One solution is to add one step to add our LLM to summarize the history and the last question, making it a better fit for our vector search query. Let's do that as a new step in our chain:

# COMMAND ----------

# DBTITLE 1,Contextual Query Generation Chain
from langchain.schema.runnable import RunnableBranch

generate_query_to_retrieve_context_template = """
Based on the chat history below, we want you to generate a query for an external data source to retrieve relevant documents so that we can better answer the question. The query should be in natual language. The external data source uses similarity search to search for relevant documents in a vector space. So the query should be similar to the relevant documents semantically. Answer with only the query. Do not add explanation.

Chat history: {chat_history}

Question: {question}
"""

generate_query_to_retrieve_context_prompt = PromptTemplate(
  input_variables= ["chat_history", "question"],
  template = generate_query_to_retrieve_context_template
)

generate_query_to_retrieve_context_chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_question),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_history),
    }
    | RunnableBranch(  #Augment query only when there is a chat history
      (lambda x: x["chat_history"], generate_query_to_retrieve_context_prompt | chat_model | StrOutputParser()),
      (lambda x: not x["chat_history"], RunnableLambda(lambda x: x["question"])),
      RunnableLambda(lambda x: x["question"])
    )
)

if debug_flag:
    #Let's try it
    output = generate_query_to_retrieve_context_chain.invoke({
        "messages": [
            {"role": "user", "content": "What is Apache Spark?"}
        ]
    })
    print(f"Test retriever query without history: {output}")

    output = generate_query_to_retrieve_context_chain.invoke({
        "messages": [
            {"role": "user", "content": "What is Apache Spark?"}, 
            {"role": "assistant", "content": "Apache Spark is an open-source data processing engine that is widely used in big data analytics."}, 
            {"role": "user", "content": "Does it support streaming?"}
        ]
    })
    print(f"Test retriever question, summarized with history: {output}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Let's put it together
# MAGIC
# MAGIC <img src="https://github.com/prasadkona/databricks_demos/blob/main/images/llm-rag-full-pinecone-7.png?raw=true" style="float: right" width="600px">
# MAGIC
# MAGIC
# MAGIC Let's now merge the retriever and the full Langchain chain.
# MAGIC
# MAGIC We will use a custom langchain template for our assistant to give proper answer.
# MAGIC
# MAGIC Make sure you take some time to try different templates and adjust your assistant tone and personality for your requirement.
# MAGIC
# MAGIC

# COMMAND ----------

from langchain.schema.runnable import RunnableBranch, RunnableParallel, RunnablePassthrough

question_with_history_and_context_str = """
You are a trustful assistant for Databricks users. You are answering python, coding, SQL, data engineering, spark, data science, AI, ML, Datawarehouse, platform, API or infrastructure, Cloud administration question related to Databricks. If you do not know the answer to a question, you truthfully say you do not know. Read the discussion to get the context of the previous conversation. In the chat discussion, you are referred to as "system". The user is referred to as "user".

Discussion: {chat_history}

Here's some context which might or might not help you answer: {context}

Answer straight, do not repeat the question, do not start with something like: the answer to the question, do not add "AI" in front of your answer, do not say: here is the answer, do not mention the context or the question.

Based on this history and context, answer this question: {question}
"""

question_with_history_and_context_prompt = PromptTemplate(
  input_variables= ["chat_history", "context", "question"],
  template = question_with_history_and_context_str
)

def format_context(docs):
    return "\n\n".join([d.page_content for d in docs])

def extract_source_urls(docs):
    return [d.metadata["url"] for d in docs]

relevant_question_chain = (
  RunnablePassthrough() |
  {
    "relevant_docs": generate_query_to_retrieve_context_prompt | chat_model | StrOutputParser() | retriever,
    "chat_history": itemgetter("chat_history"), 
    "question": itemgetter("question")
  }
  |
  {
    "context": itemgetter("relevant_docs") | RunnableLambda(format_context),
    "sources": itemgetter("relevant_docs") | RunnableLambda(extract_source_urls),
    "chat_history": itemgetter("chat_history"), 
    "question": itemgetter("question")
  }
  |
  {
    "prompt": question_with_history_and_context_prompt,
    "sources": itemgetter("sources")
  }
  |
  {
    "result": itemgetter("prompt") | chat_model | StrOutputParser(),
    "sources": itemgetter("sources")
  }
)

irrelevant_question_chain = (
  RunnableLambda(lambda x: {"result": 'I cannot answer questions that are not about Databricks.', "sources": []})
)

branch_node = RunnableBranch(
  (lambda x: "yes" in x["question_is_relevant"].lower(), relevant_question_chain),
  (lambda x: "no" in x["question_is_relevant"].lower(), irrelevant_question_chain),
  irrelevant_question_chain
)

full_chain = (
  {
    "question_is_relevant": is_about_databricks_chain,
    "question": itemgetter("messages") | RunnableLambda(extract_question),
    "chat_history": itemgetter("messages") | RunnableLambda(extract_history),    
  }
  | branch_node
)

# COMMAND ----------

# MAGIC %md 
# MAGIC Let's try our full chain:

# COMMAND ----------

if debug_flag:
  def display_chat(chat_history, response):
    def user_message_html(message):
      return f"""
        <div style="width: 90%; border-radius: 10px; background-color: #c2efff; padding: 10px; box-shadow: 2px 2px 2px #F7f7f7; margin-bottom: 10px; font-size: 14px;">
          {message}
        </div>"""
    def assistant_message_html(message):
      return f"""
        <div style="width: 90%; border-radius: 10px; background-color: #e3f6fc; padding: 10px; box-shadow: 2px 2px 2px #F7f7f7; margin-bottom: 10px; margin-left: 40px; font-size: 14px">
          <img style="float: left; width:40px; margin: -10px 5px 0px -10px" src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/robot.png?raw=true"/>
          {message}
        </div>"""
    chat_history_html = "".join([user_message_html(m["content"]) if m["role"] == "user" else assistant_message_html(m["content"]) for m in chat_history])
    answer = response["result"].replace('\n', '<br/>')
    sources_html = ("<br/><br/><br/><strong>Sources:</strong><br/> <ul>" + '\n'.join([f"""<li><a href="{s}">{s}</a></li>""" for s in response["sources"]]) + "</ul>") if response["sources"] else ""
    response_html = f"""{answer}{sources_html}"""

    displayHTML(chat_history_html + assistant_message_html(response_html))

# COMMAND ----------

# DBTITLE 1,Asking an out-of-scope question
import json

if debug_flag:
    non_relevant_dialog = {
        "messages": [
            {"role": "user", "content": "What is Apache Spark?"}, 
            {"role": "assistant", "content": "Apache Spark is an open-source data processing engine that is widely used in big data analytics."}, 
            {"role": "user", "content": "Why is the sky blue?"}
        ]
    }
    print(f'Testing with a non relevant question...')
    response = full_chain.invoke(non_relevant_dialog)
    display_chat(non_relevant_dialog["messages"], response)

# COMMAND ----------

# DBTITLE 1,Asking a relevant question
if debug_flag:
    dialog = {
        "messages": [
            {"role": "user", "content": "What is Apache Spark?"}, 
            {"role": "assistant", "content": "Apache Spark is an open-source data processing engine that is widely used in big data analytics."}, 
            {"role": "user", "content": "Does it support streaming?"},
            {"role": "assistant", "content": "Yes."},
            {"role": "user", "content": "Tell me more about it's capabilities."},
        ]
    }
    print(f'Testing with relevant history and question...')
    response = full_chain.invoke(dialog)
    display_chat(dialog["messages"], response)

# COMMAND ----------

# DBTITLE 1,Setting the full chain for logging with mlfow
import mlflow

mlflow.models.set_model(model=full_chain)


# COMMAND ----------

# DBTITLE 1,Testing the chain with a input that includes history
if debug_flag:
  #Get our model signature from input/output
  from mlflow.models import infer_signature
  output = full_chain.invoke(dialog)
  signature = infer_signature(dialog, output)
  print("output.......")
  print( output)
  print("signature.......")
  print(signature)

# COMMAND ----------

# DBTITLE 1,Testing the chain with a input that doesnot include history
if debug_flag:
  #Get our model signature from input/output
  from mlflow.models import infer_signature
  input_example = {
   "messages": [
       {
           "role": "user",
           "content": "How does billing work on Databricks?",
       }
   ]
  }
  output = full_chain.invoke(input_example)
  print("output.......")
  print( output)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Next we register this Rag chatbot chain with Databricks unity catalog and then deploy to model serving
