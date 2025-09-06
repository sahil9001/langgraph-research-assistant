import operator
from typing import List
from typing_extensions import TypedDict
from classes.analyst import Analyst
from typing import Annotated
from langgraph.graph import MessagesState

class GenerateAnalystsState(TypedDict):
    topic: str
    max_analysts: int
    human_analyst_feedback: str
    analysts: List[Analyst]


class InterviewState(MessagesState):
    max_num_turns: int
    context: Annotated[list, operator.add]
    analyst: Analyst
    interview: str
    sections: list

