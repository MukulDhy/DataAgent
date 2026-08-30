import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from utils.llm_pick import pick_llm
from Models.schema import AgentSchema

def curate_question(state : AgentSchema ) -> AgentSchema: 
    llmObject = pick_llm("low")
    user_question = state.user_ques
    response  = llmObject.invoke(f"Can you Curract this Question for me {user_question}")
    state.curated_ques = response.content
    return state



    