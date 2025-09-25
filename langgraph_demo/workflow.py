"""LangGraph workflow definition for the demo application."""
from typing import Annotated, Dict, List, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from .config import get_settings

# Define the state structure
class WorkflowState(TypedDict):
    """State for the LangGraph workflow."""
    messages: Annotated[List, add_messages]
    response: str


class LangGraphWorkflow:
    """LangGraph workflow for the demo application."""

    def __init__(self):
        """Initialize the workflow with settings."""
        self.settings = get_settings()
        self.llm = ChatOpenAI(
            model=self.settings.model_name,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
            api_key=self.settings.openai_api_key,
        )
        self.workflow = self._build_workflow()

    def _build_workflow(self):
        """Build and return the LangGraph workflow."""
        # Define the nodes
        def generate_response(state: WorkflowState) -> Dict[str, str]:
            """Generate a response using the LLM."""
            messages = state["messages"]
            response = self.llm.invoke(messages)
            ai_message = AIMessage(content=response.content)
            return {"response": response.content, "messages": [ai_message]}

        # Define the graph
        workflow = StateGraph(WorkflowState)

        # Add nodes
        workflow.add_node("generate", generate_response)

        # Set the entry point
        workflow.set_entry_point("generate")

        # Connect the nodes
        workflow.add_edge("generate", END)

        # Compile the workflow
        return workflow.compile()

    def process_message(self, message: str) -> str:
        """
        Process a message through the workflow.

        Args:
            message: The user's message

        Returns:
            str: The generated response
        """
        if message is None or not str(message).strip():
            raise ValueError("Input message must be a non-empty string.")
        # Prepare the initial state
        system_message = SystemMessage(
            content="""You are a helpful AI assistant. Keep your responses concise and helpful.
            If you don't know the answer, say you don't know."""
        )
        human_message = HumanMessage(content=message)

        # Run the workflow
        result = self.workflow.invoke(
            {"messages": [system_message, human_message], "response": ""}
        )

        return result["response"]
