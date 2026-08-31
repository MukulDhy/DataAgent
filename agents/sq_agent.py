import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from utils.llm_pick import pick_llm
from Models.schema import AgentSchema
from langchain_core.messages import HumanMessage,SystemMessage
from utils.database import getSchemaDetails


def curate_question(state : AgentSchema ) -> AgentSchema: 
    llmObject = pick_llm("low")
    user_question = state.user_ques  
    response  = llmObject.invoke(f"Can you Curract this Question for me {user_question}")
    state.curated_ques = response.content
    state.messages = state.messages + [HumanMessage(content = f"{response}")]
    return state

def prompt_query_context(state : AgentSchema ) -> AgentSchema:
    
    schema_details = getSchemaDetails()
    user_query = state.curated_ques
    # Constructing the prompt query for the agent to generate the SQL query
    prompt = f"""
    You are an SQL Analyst Agent. Your task is to convert the user's natural language
    query into a PostgreSQL SQL query that can be executed directly on the database.

    You are provided with:
    - The user's original query.
    - The database schema details, including table names, column names, data types,
    and sample data for each table.

    Use the provided schema and sample data to understand the database structure and
    generate an accurate SQL query.

    Rules:
    1. Generate only valid PostgreSQL SQL.
    2. Unless the user explicitly asks for a specific number of rows, always limit
    the output to 10 rows using LIMIT 10.
    3. If the user's query already specifies a number of rows, follow the user's
    requested limit instead.
    4. Do not invent table names, column names, or relationships that are not present
    in the provided schema.
    5. Use appropriate PostgreSQL syntax.
    6. Do not include explanations, comments, markdown, code fences, or any additional
    text.
    7. The generated SQL query will be executed directly on the database, so it must
    be ready to execute without any modifications.

    User's Original Query:
    {user_query}

    Database Schema Details:
    {schema_details}
    """
    
    state.messages = state.messages + [SystemMessage(content=prompt)]
    state.prompt_query_context = prompt
    
    llmObject = pick_llm("high")
    gernate_sql_query   = llmObject.invoke(prompt)
    state.generated_sql_query = gernate_sql_query
    return state
    



    