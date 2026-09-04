import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.llm_pick import pick_llm
from Models.schema import AgentSchema, JudgeSchema
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from utils.database import getSchemaDetails, getDatabaseObject
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display


def curate_question(state: AgentSchema) -> AgentSchema:

    user_question = state.user_question

    try:
        # First try LOW model
        llmObject = pick_llm("low")

        response = llmObject.invoke(
            f"""
            Correct and clarify this question without changing its meaning.

            Question:
            {user_question}

            Return only the corrected question.
            """
        )

    except Exception:
        try:
            # Fallback to MEDIUM model
            llmObject = pick_llm("medium")

            response = llmObject.invoke(
                f"""
                Correct and clarify this question without changing its meaning.

                Question:
                {user_question}

                Return only the corrected question.
                """
            )

        except Exception:
            # Final fallback
            response = None

    # If both models failed, use original question
    if response is None or not response.content.strip():
        curated_question = user_question
    else:
        curated_question = response.content.strip()

    state.curated_ques = curated_question
    print(state.curated_ques)
    state.messages = state.messages + [HumanMessage(content=curated_question)]

    return state


def prompt_query_context(state: AgentSchema) -> AgentSchema:

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
    gernate_sql_query = llmObject.invoke(prompt).content
    state.generated_sql_query = gernate_sql_query
    print(state.generated_sql_query)
    return state


def is_safe_sql(state: AgentSchema) -> AgentSchema:

    sql_query = state.generated_sql_query
    llm = pick_llm("high")
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

    response = llm_judge.invoke(prompt.format(sql_query=sql_query)).model_dump()

    if response["answer"] == "Yes":
        state.is_safe = "Yes"
    else:
        state.is_safe = "No"

    return state


def execute_query(state: AgentSchema) -> AgentSchema:
    sql_query = state.generated_sql_query

    obj = getDatabaseObject()

    state.sql_query_execution_result = obj.execute_sql(sql_query)

    return state


# Canceled SQL Query Node
def canceled_sql(state: AgentSchema) -> AgentSchema:

    comments = state.comments

    state.final_answer = f"The generated SQL query was deemed unsafe to execute. The reason provided by the judge is: {comments}. Therefore, the SQL query will not be executed."
    state.messages = state.messages + [AIMessage(content=f"{state.final_answer}")]

    return state


def represent_final_answer(state: AgentSchema) -> AgentSchema:

    execution_result = state.sql_query_execution_result
    curated_question = state.curated_ques

    llm = pick_llm("medium")

    prompt = f"""
    You are an SQL analyst agent. Your task is to provide a final answer to the user based on the
    execution result of the SQL query and the user's original question. The final answer should be
    concise, clear, and directly address the user's query. Avoid including any SQL code or technical
    details in the final answer. The final answer should be in a user-friendly format that is easy to
    understand. If the execution result is empty or does not provide a clear answer to the user's question, explain this in the final answer. \n
    Here is the execution result: {execution_result} \n
    Here is the user's original question: {curated_question}
    """

    llm_response = llm.invoke(prompt).content  # Get the final answer from the LLM

    state.final_answer = llm_response
    state.messages = state.messages + [
        AIMessage(content=f"{llm_response}")
    ]  # Append the final answer to the messages list

    return state


sql_agent_graph = StateGraph(AgentSchema)


def drawStateGraph():
    import tkinter as tk
    from PIL import Image, ImageTk

    sql_agent = sql_agent_graph.compile()

    png_bytes = sql_agent.get_graph().draw_mermaid_png()

    with open("temp_graph.png", "wb") as f:
        f.write(png_bytes)

    root = tk.Tk()
    root.title("SQL Agent Graph")

    image = Image.open("temp_graph.png")
    photo = ImageTk.PhotoImage(image)

    label = tk.Label(root, image=photo)
    label.pack()

    root.mainloop()


# Codintional Edge Function
def is_safe_sql_edge(state: AgentSchema) -> str:
    is_safe = state.is_safe

    if is_safe.lower() == "yes":
        return "execute_query"

    else:
        return "canceled_sql"


def buildStateGraph():

    sql_agent_graph.add_node(curate_question, name="curate_ques")
    sql_agent_graph.add_node(
        prompt_query_context, name="User Natural Language to SQL Query"
    )
    sql_agent_graph.add_node(is_safe_sql, name="Sql Safe Judge")
    sql_agent_graph.add_node(execute_query, name="Execute Query of SQL")
    sql_agent_graph.add_node(canceled_sql, name="Canceled Query")
    sql_agent_graph.add_node(represent_final_answer, name="Repersent the Final Answer")

    # Add Edges
    sql_agent_graph.add_edge(START, "curate_question")
    sql_agent_graph.add_edge("curate_question", "prompt_query_context")
    sql_agent_graph.add_edge("prompt_query_context", "is_safe_sql")

    sql_agent_graph.add_conditional_edges(
        "is_safe_sql",
        is_safe_sql_edge,
        {"execute_query": "execute_query", "canceled_sql": "canceled_sql"},
    )
    sql_agent_graph.add_edge("canceled_sql", END)
    sql_agent_graph.add_edge("execute_query", "represent_final_answer")
    sql_agent_graph.add_edge("represent_final_answer", END)
    # sql_agent_graph.add_edge(START)

    # sql_agent = sql_agent_graph.compile()

    # display(Image(sql_agent.get_graph().draw_mermaid_png()))


def main():
    print("Hey There : Ask me Any Database Query \n")
    # user_asked = input("")
    # print("User Query = ", user_asked)
    buildStateGraph()

    sql_analyst = sql_agent_graph.compile()
    input_schema = AgentSchema(
        messages=[],
        user_question="fetch the deialt of the mukul recors all",
        curated_ques="",
        prompt_query_context="",
        generated_sql_query="",
        is_safe="No",
        comments="",
        sql_query_execution_result="",
        final_answer="",
    )

    # Execute the Graph
    sql_analyst_response = sql_analyst.invoke(input_schema)
    # print(
    #     sql_analyst_response["messages"]
    # )  # Print the final output of the graph execution
    # print("********************************")

    # print(sql_analyst_response["generated_sql_query"])  # Print the generated SQL query

    # print("********************************")

    # print(
    #     sql_analyst_response["sql_query_execution_result"]
    # )  # Print the result of executing the SQL query

    # print("********************************")

    # print(
    #     sql_analyst_response["prompt_query_context"]
    # )  # Print the prompt query context

    print(sql_analyst_response["final_answer"])  # Print the prompt query context


if __name__ == "__main__":
    main()
