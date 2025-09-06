"""
LangGraph Research Assistant
A multi-agent system for conducting research interviews and generating comprehensive reports.
"""

import os 
from langchain_ollama import ChatOllama
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.messages import get_buffer_string
from langgraph.types import Send

from classes.state import GenerateAnalystsState, InterviewState, ResearchGraphState
from classes.analyst import Perspectives, SearchQuery
from prompts.prompts import (
    analyst_instructions, question_instructions, search_instructions, 
    answer_instructions, section_writer_instructions, report_writer_instructions, 
    intro_conclusion_instructions
)
from dotenv import load_dotenv

load_dotenv()

os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["LANGSMITH_TRACING"] = 'true'
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")


# Initialize LLM and search tools
llm = ChatOllama(model="deepseek-r1", temperature=0)

# Initialize Tavily search (optional - requires TAVILY_API_KEY environment variable)
try:
    tavily_search = TavilySearchResults(max_results=3)
    TAVILY_AVAILABLE = True
except Exception as e:
    print(f"⚠️  Tavily search not available: {e}")
    print("   To enable web search, set the TAVILY_API_KEY environment variable")
    tavily_search = None
    TAVILY_AVAILABLE = False


# =============================================================================
# ANALYST GENERATION FUNCTIONS
# =============================================================================

def create_analysts(state: GenerateAnalystsState):
    """Generate a set of diverse analysts for the research topic."""
    topic = state['topic']
    max_analysts = state['max_analysts']
    human_analyst_feedback = state.get('human_analyst_feedback', None)

    # Use structured output to ensure proper format
    structured_llm = llm.with_structured_output(Perspectives)
    system_message = analyst_instructions.format(
        topic=topic, 
        max_analysts=max_analysts, 
        human_analyst_feedback=human_analyst_feedback
    )

    analysts = structured_llm.invoke([
        SystemMessage(content=system_message), 
        HumanMessage(content="Generate a set of analysts.")
    ])

    return {"analysts": analysts.analysts}

def human_feedback(state: GenerateAnalystsState):
    """Placeholder for human feedback collection."""
    print("👤 Human feedback node - no feedback provided, continuing...")
    pass

def should_continue(state: GenerateAnalystsState):
    """Determine whether to continue with analyst generation or end."""
    human_analyst_feedback = state.get("human_analyst_feedback", None)
    return "create_analysts" if human_analyst_feedback else END


# =============================================================================
# INTERVIEW CONDUCTING FUNCTIONS
# =============================================================================

def generate_question(state: InterviewState):
    """Generate a question for the analyst based on their persona and conversation history."""
    analyst = state['analyst']
    messages = state['messages']

    print(f"❓ Generating question for {analyst.name}...")

    system_message = question_instructions.format(goals=analyst.persona)
    question = llm.invoke([SystemMessage(content=system_message)] + messages)

    print(f"✅ Generated question for {analyst.name}")
    return {"messages": [question]}

def search_web(state: InterviewState):
    """Retrieve documents from web search using Tavily."""
    print("🔍 Searching web...")
    if not TAVILY_AVAILABLE:
        print("⚠️  Web search not available")
        return {"context": ["Web search not available - Tavily API key not configured"]}
    
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([
        SystemMessage(content=search_instructions)
    ] + state['messages'])

    print(f"🔍 Web search query: {search_query.search_query}")
    search_docs = tavily_search.invoke(search_query.search_query)

    formatted_search_docs = "\n\n---\n\n".join([
        f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
        for doc in search_docs
    ])

    print(f"✅ Found {len(search_docs)} web documents")
    return {"context": [formatted_search_docs]}

def search_wikipedia(state: InterviewState):
    """Retrieve documents from Wikipedia."""
    print("📚 Searching Wikipedia...")
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([
        SystemMessage(content=search_instructions)
    ] + state['messages'])

    print(f"📚 Wikipedia search query: {search_query.search_query}")
    search_docs = WikipediaLoader(query=search_query.search_query, load_max_docs=2).load()

    formatted_search_docs = "\n\n---\n\n".join([
        f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page","")}"/>\n{doc.page_content}\n</Document>'
        for doc in search_docs
    ])

    print(f"✅ Found {len(search_docs)} Wikipedia documents")
    return {"context": [formatted_search_docs]}

def generate_answer(state: InterviewState):
    """Generate an answer from the analyst using retrieved context."""
    analyst = state['analyst']
    messages = state['messages']
    context = state['context']

    system_message = answer_instructions.format(goals=analyst.persona, context=context)
    answer = llm.invoke([SystemMessage(content=system_message)] + messages)
    answer.name = 'expert'

    return {'messages': [answer]}

def save_interview(state: InterviewState):
    """Save the interview conversation as a string."""
    messages = state['messages']
    interview = get_buffer_string(messages)
    print(f"💾 Saving interview with {len(messages)} messages")
    return {"interview": interview}

def route_messages(state: InterviewState, name: str = 'expert'):
    """Route between question generation and interview completion."""
    messages = state['messages']
    max_num_turns = state.get('max_num_turns', 2)

    num_responses = len([
        m for m in messages if isinstance(m, AIMessage) and m.name == name
    ])

    print(f"🔄 Routing: {num_responses}/{max_num_turns} responses from {name}")

    # End if expert has answered more than the max turns
    if num_responses >= max_num_turns:
        print("✅ Interview complete - max turns reached")
        return 'save_interview'
    
    last_question = messages[-2]
    if "Thank you so much for your help" in last_question.content:
        print("✅ Interview complete - thank you message detected")
        return 'save_interview'
    
    print("❓ Continuing interview - asking next question")
    return 'ask_question'

def write_section(state: InterviewState):
    """Write a section based on the interview and context."""
    interview = state['interview']
    context = state['context']
    analyst = state["analyst"]

    print(f"📝 Writing section for {analyst.name}...")

    system_message = section_writer_instructions.format(focus=analyst.description)
    section = llm.invoke([
        SystemMessage(content=system_message), 
        HumanMessage(content=f"Use this source to write your section: {context}")
    ])

    print(f"✅ Completed section for {analyst.name}")
    return {"sections": [section.content]}


# =============================================================================
# REPORT GENERATION FUNCTIONS
# =============================================================================

def write_report(state: ResearchGraphState):
    """Generate the main report content from all interview sections."""
    sections = state["sections"]
    topic = state["topic"]

    print(f"📊 Writing main report from {len(sections)} sections...")

    # Concatenate all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    
    # Generate the main report content
    system_message = report_writer_instructions.format(topic=topic, context=formatted_str_sections)    
    report = llm.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content="Write a report based upon these memos.")
    ]) 
    
    print("✅ Completed main report content")
    return {"content": report.content}

def write_introduction(state: ResearchGraphState):
    """Generate the report introduction based on all sections."""
    sections = state["sections"]
    topic = state["topic"]

    # Concatenate all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    
    # Generate introduction
    instructions = intro_conclusion_instructions.format(
        topic=topic, 
        formatted_str_sections=formatted_str_sections
    )    
    intro = llm.invoke([
        SystemMessage(content=instructions),
        HumanMessage(content="Write the report introduction")
    ]) 
    return {"introduction": intro.content}

def write_conclusion(state: ResearchGraphState):
    """Generate the report conclusion based on all sections."""
    sections = state["sections"]
    topic = state["topic"]

    # Concatenate all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    
    # Generate conclusion
    instructions = intro_conclusion_instructions.format(
        topic=topic, 
        formatted_str_sections=formatted_str_sections
    )    
    conclusion = llm.invoke([
        SystemMessage(content=instructions),
        HumanMessage(content="Write the report conclusion")
    ]) 
    return {"conclusion": conclusion.content}

def finalize_report(state: ResearchGraphState):
    """Combine all report components into a final comprehensive report."""
    content = state["content"]
    
    # Clean up content formatting
    if content.startswith("## Insights"):
        content = content.strip("## Insights")
    
    # Extract sources if present
    if "## Sources" in content:
        try:
            content, sources = content.split("\n## Sources\n")
        except:
            sources = None
    else:
        sources = None

    # Combine all components
    final_report = (state["introduction"] + "\n\n---\n\n" + 
                   content + "\n\n---\n\n" + state["conclusion"])
    
    if sources is not None:
        final_report += "\n\n## Sources\n" + sources
        
    return {"final_report": final_report}

def initiate_all_interviews(state: ResearchGraphState):
    """Initiate parallel interviews for all analysts using the Send API."""
    # Check if human feedback is needed
    human_analyst_feedback = state.get('human_analyst_feedback')
    if human_analyst_feedback:
        print("🔄 Returning to analyst creation due to human feedback")
        return "create_analysts"

    # Otherwise kick off interviews in parallel via Send() API
    topic = state["topic"]
    analysts = state["analysts"]
    print(f"🚀 Initiating parallel interviews for {len(analysts)} analysts...")
    
    for i, analyst in enumerate(analysts):
        print(f"  {i+1}. Starting interview with {analyst.name}")
    
    return [
        Send("conduct_interview", {
            "analyst": analyst,
            "messages": [HumanMessage(content=f"So you said you were writing an article on {topic}?")]
        }) 
        for analyst in analysts
    ]


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def build_analyst_generation_graph():
    """Build the analyst generation subgraph."""
    builder = StateGraph(GenerateAnalystsState)
    builder.add_node("create_analysts", create_analysts)
    builder.add_node("human_feedback", human_feedback)
    builder.add_edge(START, "create_analysts")
    builder.add_edge("create_analysts", "human_feedback")
    builder.add_conditional_edges("human_feedback", should_continue, ["create_analysts", END])

    memory = MemorySaver()
    return builder.compile(interrupt_before=['human_feedback'], checkpointer=memory)

def build_interview_graph():
    """Build the interview conducting subgraph."""
    interview_builder = StateGraph(InterviewState)
    interview_builder.add_node("ask_question", generate_question)
    interview_builder.add_node("search_web", search_web)
    interview_builder.add_node("search_wikipedia", search_wikipedia)
    interview_builder.add_node("answer_question", generate_answer)
    interview_builder.add_node("save_interview", save_interview)
    interview_builder.add_node("write_section", write_section)

    # Define the flow
    interview_builder.add_edge(START, "ask_question")
    interview_builder.add_edge("ask_question", "search_web")
    interview_builder.add_edge("ask_question", "search_wikipedia")
    interview_builder.add_edge("search_web", "answer_question")
    interview_builder.add_edge("search_wikipedia", "answer_question")
    interview_builder.add_conditional_edges("answer_question", route_messages, ['ask_question', 'save_interview'])
    interview_builder.add_edge("save_interview", "write_section")
    interview_builder.add_edge("write_section", END)

    interview_memory = MemorySaver()
    return interview_builder.compile(checkpointer=interview_memory).with_config(run_name="Conduct Interviews")

def build_main_research_graph():
    """Build the main research graph that orchestrates the entire process."""
    builder = StateGraph(ResearchGraphState)
    builder.add_node("create_analysts", create_analysts)
    builder.add_node("human_feedback", human_feedback)
    builder.add_node("conduct_interview", build_interview_graph())
    builder.add_node("write_report", write_report)
    builder.add_node("write_introduction", write_introduction)
    builder.add_node("write_conclusion", write_conclusion)
    builder.add_node("finalize_report", finalize_report)

    # Define the flow
    builder.add_edge(START, "create_analysts")
    builder.add_edge("create_analysts", "human_feedback")
    builder.add_conditional_edges("human_feedback", initiate_all_interviews, ["create_analysts", "conduct_interview"])
    
    # After all interviews are complete, run report generation in parallel
    builder.add_edge("conduct_interview", "write_report")
    builder.add_edge("conduct_interview", "write_introduction") 
    builder.add_edge("conduct_interview", "write_conclusion")
    
    # Wait for all report components to complete before finalizing
    builder.add_edge(["write_conclusion", "write_report", "write_introduction"], "finalize_report")
    builder.add_edge("finalize_report", END)

    # Compile with memory
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

# Build the main research graph
graph = build_main_research_graph()


# Main execution
if __name__ == "__main__":
    # Configuration
    max_analysts = 3
    topic = '''The benefits of adopting LangGraph as an agent framework.'''
    thread = {"configurable": {"thread_id": "research_session"}}

    print("🚀 Starting Research Assistant...")
    print(f"📝 Topic: {topic}")
    print(f"👥 Max Analysts: {max_analysts}")
    print("-" * 60)

    # Step 1: Generate analysts
    print("\n🔍 Step 1: Generating analysts...")
    for event in graph.stream({"topic": topic, "max_analysts": max_analysts}, thread, stream_mode="values"):
        analysts = event.get("analysts", [])
        if analysts:
            print(f"✅ Generated {len(analysts)} analysts:")
            for i, analyst in enumerate(analysts, 1):
                print(f"  {i}. {analyst.name} ({analyst.role})")
                print(f"     Affiliation: {analyst.affiliation}")
                print(f"     Focus: {analyst.description}")
                print()

    # Get final analysts
    final_state = graph.get_state(thread)
    analysts = final_state.values.get('analysts', [])
    
    if not analysts:
        print("❌ No analysts generated. Cannot proceed with interviews.")
        exit(1)

    # Step 2: Conduct parallel interviews
    print(f"\n🎯 Step 2: Conducting parallel interviews with {len(analysts)} analysts...")
    
    # Run the full research graph with parallelization
    research_thread = {"configurable": {"thread_id": "full_research"}}
    
    print("📊 Running full research pipeline...")
    for event in graph.stream({
        "topic": topic, 
        "max_analysts": max_analysts
        # Don't pass analysts - let the graph generate them
    }, research_thread, stream_mode="values"):
        
        # Show progress for different stages
        if "sections" in event and event["sections"]:
            print(f"✅ Generated {len(event['sections'])} interview sections")
            print(f"   Sections: {[s[:100] + '...' if len(s) > 100 else s for s in event['sections']]}")
        
        if "content" in event and event["content"]:
            print("✅ Generated main report content")
            
        if "introduction" in event and event["introduction"]:
            print("✅ Generated report introduction")
            
        if "conclusion" in event and event["conclusion"]:
            print("✅ Generated report conclusion")
            
        if "final_report" in event and event["final_report"]:
            print("✅ Finalized complete report")
            
        # Debug: Show all keys in the event
        if event:
            print(f"🔍 Event keys: {list(event.keys())}")
            
        # Debug: Show the full state to understand what's happening
        print(f"🔍 Full state keys: {list(event.keys())}")
        for key, value in event.items():
            if key not in ['topic', 'max_analysts', 'analysts']:
                print(f"   {key}: {type(value)} - {str(value)[:200]}...")

    # Get final results
    final_research_state = graph.get_state(research_thread)
    final_report = final_research_state.values.get('final_report', '')
    
    if final_report:
        print("\n" + "="*80)
        print("📋 FINAL RESEARCH REPORT")
        print("="*80)
        print(final_report)
        print("="*80)
    else:
        print("❌ Failed to generate final report")
        
        # Debug: Let's try running a single interview to see what's happening
        print("\n🔍 Debug: Testing single interview...")
        interview_graph = build_interview_graph()
        test_analyst = final_research_state.values.get('analysts', [])[0] if final_research_state.values.get('analysts') else None
        
        if test_analyst:
            print(f"🧪 Testing interview with {test_analyst.name}")
            test_thread = {"configurable": {"thread_id": "test_interview"}}
            test_result = interview_graph.invoke({
                "analyst": test_analyst,
                "messages": [HumanMessage(content=f"So you said you were writing an article on {topic}?")],
                "max_num_turns": 2
            }, test_thread)
            print(f"✅ Test interview result keys: {list(test_result.keys())}")
            if "sections" in test_result:
                print(f"✅ Test interview generated {len(test_result['sections'])} sections")
        else:
            print("❌ No analysts available for testing")

