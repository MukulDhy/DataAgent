import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.llm_pick import pick_llm
from Models.schema import AgentSchema, JudgeSchema
from langchain_core.messages import HumanMessage, SystemMessage
from utils.database import getSchemaDetails


def is_safe_sql(state : AgentSchema) -> AgentSchema:
    
    sql_query = state.generated_sql_query      
    llm = pick_llm("medium")
    llm_judge = llm.with_structured_output(schema=JudgeSchema)

    prompt = """
    You are a SQL security judge.

    Your task is to determine whether the given SQL query is SAFE to execute.

    A query is considered SAFE only if it is strictly READ-ONLY.

    SAFE queries:
    - SELECT
    - SELECT with JOIN
    - SELECT with WHERE, GROUP BY, HAVING, ORDER BY, LIMIT, OFFSET
    - SELECT with CTEs (WITH ...)
    - SELECT using aggregate functions such as COUNT, SUM, AVG, MIN, MAX
    - SELECT using window functions
    - EXPLAIN or other read-only query-planning statements, if supported
    - Read-only queries that access database metadata/information_schema

    UNSAFE queries:
    - INSERT
    - UPDATE
    - DELETE
    - DROP
    - ALTER
    - TRUNCATE
    - CREATE
    - GRANT
    - REVOKE
    - MERGE
    - REPLACE
    - COPY that writes data
    - CALL or procedures/functions that may modify database state
    - Any transaction/control statement that can modify database state
    - Multiple SQL statements where ANY statement is unsafe
    - SQL containing comments or obfuscation intended to hide a write operation
    - Any query whose safety cannot be confidently determined

    IMPORTANT RULES:
    1. The query must not modify, delete, insert, update, create, alter, or otherwise change database state.
    2. If the query contains multiple statements, mark it "No" if even one statement is unsafe.
    3. Do not assume that a function is safe merely because it appears inside a SELECT. A SELECT that calls a function capable of modifying database state must be considered unsafe.
    4. When in doubt, return "No".
    5. Do not execute the query. Only analyze it.
    6. Return "Yes" only when you are confident that the query is read-only.

    Analyze the following SQL query:

    {sql_query}
    """

    sql_query = """
    SELECT customer_id, COUNT(*) AS total_orders
    FROM orders
    GROUP BY customer_id
    ORDER BY total_orders DESC;
    """

    response = llm_judge.invoke(prompt.format(sql_query=sql_query)).model_dump()

    if response['answer'] == "Yes" :
        state.is_safe = "Yes"
    else:
        state.is_safe = "No"
        
    return state
        