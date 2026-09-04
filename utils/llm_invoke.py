import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
import llm_pick
from Models.schema import AgentSchema

def llm_call(level:str,message:AgentSchema):
    
    llm_object = llm_pick(level)
    llm_response = llm_object.invoke(message)
    print(llm_response.content)