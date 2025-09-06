import os 
from langchain_ollama import ChatOllama
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.messages import get_buffer_string
from classes.state import GenerateAnalystsState, InterviewState
from classes.analyst import Perspectives, SearchQuery
from prompts.prompts import analyst_instructions, question_instructions, search_instructions, answer_instructions, section_writer_instructions
from dotenv import load_dotenv

load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = 'research-assistant'

# Initialize LLM
llm = ChatOllama(model="deepseek-r1", temperature=0)

# Initialize search tools
tavily_search = TavilySearchResults(max_results=3)


def create_analysts(state: GenerateAnalystsState):
    """ Function to create analysts """
    topic = state['topic']
    max_analysts=state['max_analysts']
    human_analyst_feedback=state.get('human_analyst_feedback',None)

    # Enforce the structured output
    structured_llm = llm.with_structured_output(Perspectives)

    # System message
    system_message = analyst_instructions.format(topic=topic,max_analysts=max_analysts, human_analyst_feedback=human_analyst_feedback)

    # Generate question
    analysts = structured_llm.invoke([SystemMessage(content=system_message)] + [HumanMessage(content="Generate a set of analysts.")])

    return {"analysts": analysts.analysts}

def human_feedback(state: GenerateAnalystsState):
    pass

def should_continue(state: GenerateAnalystsState):

    human_analyst_feedback=state.get("human_analyst_feedback", None)

    if human_analyst_feedback:
        return "create_analysts"
    
    return END


def generate_question(state: InterviewState):
    analyst = state['analyst']
    messages = state['messages']

    system_message = question_instructions.format(goals=analyst.persona)
    question = llm.invoke([SystemMessage(content=system_message)] + messages)

    return {"messages": [question]}


def search_web(state: InterviewState):
    """Retrieve docs from the web search"""

    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([SystemMessage(content=search_instructions)] + state['messages'])

    search_docs = tavily_search.invoke(search_query.search_query)

    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]}


def search_wikipedia(state: InterviewState):
    """ Retrieve docs from wikipedia """

    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([SystemMessage(content=search_instructions)] + state['messages'])

    search_docs = WikipediaLoader(query=search_query.search_query, load_max_docs=2).load()

    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page","")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]}


def generate_answer(state: InterviewState):
    analyst = state['analyst']
    messages = state['messages']
    context = state['context']

    system_message = answer_instructions.format(goals=analyst.persona, context=context)

    answer = llm.invoke([SystemMessage(content=system_message)] + messages)

    answer.name = 'expert'

    return {'messages': [answer]}


def save_interview(state: InterviewState):
    messages = state['messages']
    interview = get_buffer_string(messages)
    return {"interview": interview}


def route_messages(state: InterviewState, name: str = 'expert'):
    """ Route between question and answer """
    messages = state['messages']
    max_num_turns = state.get('max_num_turns', 2)

    num_responses = len(
        [m for m in messages if isinstance(m, AIMessage) and m.name == name]
    )

    # End if expert has answered more than the max turns
    if num_responses >= max_num_turns:
        return 'save_interview'
    
    last_question = messages[-2]

    if "Thank you so much for your help" in last_question.content:
        return 'save_interview'
    
    return 'ask_question'


def write_section(state: InterviewState):
    interview = state['interview']
    context = state['context']
    analyst = state["analyst"]

    system_message = section_writer_instructions.format(focus=analyst.description)

    section = llm.invoke(
        [SystemMessage(content=system_message)] + [HumanMessage(content=f"Use this source to write your section: {context}")]
    )

    return {"sections": [section.content]}


# Build the analyst generation graph
builder = StateGraph(GenerateAnalystsState)
builder.add_node("create_analysts", create_analysts)
builder.add_node("human_feedback", human_feedback)
builder.add_edge(START, "create_analysts")
builder.add_edge("create_analysts", "human_feedback")
builder.add_conditional_edges("human_feedback", should_continue, ["create_analysts", END])

memory = MemorySaver()
graph = builder.compile(interrupt_before=['human_feedback'], checkpointer=memory)

# Build the interview graph
interview_builder = StateGraph(InterviewState)
interview_builder.add_node("ask_question", generate_question)
interview_builder.add_node("search_web", search_web)
interview_builder.add_node("search_wikipedia", search_wikipedia)
interview_builder.add_node("answer_question", generate_answer)
interview_builder.add_node("save_interview", save_interview)
interview_builder.add_node("write_section", write_section)

# Flow (edges)
interview_builder.add_edge(START, "ask_question")
interview_builder.add_edge("ask_question", "search_web")
interview_builder.add_edge("ask_question", "search_wikipedia")
interview_builder.add_edge("search_web", "answer_question")
interview_builder.add_edge("search_wikipedia", "answer_question")
interview_builder.add_conditional_edges("answer_question", route_messages, ['ask_question', 'save_interview'])
interview_builder.add_edge("save_interview", "write_section")
interview_builder.add_edge("write_section", END)

interview_memory = MemorySaver()
interview_graph = interview_builder.compile(checkpointer=interview_memory).with_config(run_name="Conduct Interviews")

# Main execution
if __name__ == "__main__":
    # Generate analysts
    max_analysts = 3
    topic = '''The benefits of adopting LangGraph as an agent framework.'''
    thread = {"configurable": {"thread_id": "1"}}

    print("Generating analysts...")
    for event in graph.stream({"topic": topic, "max_analysts": max_analysts,}, thread, stream_mode="values"):
        analysts = event.get("analysts",'')
        if analysts:
            for analyst in analysts: 
                print(f"Name: {analyst.name}")
                print(f"Affiliation: {analyst.affiliation}")
                print(f"Role: {analyst.role}")
                print(f"Description: {analyst.description}")
                print("-"*50)

    # Get final analysts
    final_state = graph.get_state(thread)
    analysts = final_state.values.get('analysts', [])
    
    if analysts:
        print(f"\nConducting interview with {analysts[0].name}...")
        messages = [HumanMessage(f"So you said you were writing an article on {topic}?")]
        interview_thread = {"configurable": {"thread_id": "2"}}
        
        interview = interview_graph.invoke({"analyst": analysts[0], "messages": messages, "max_num_turns": 2}, interview_thread)
        print("\nGenerated section:")
        print(interview['sections'][0])
    else:
        print("No analysts generated. Cannot conduct interview.")