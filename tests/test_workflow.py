"""Tests for the LangGraph workflow."""
import pytest
from unittest.mock import MagicMock, patch

from langgraph_demo.workflow import LangGraphWorkflow, WorkflowState


class TestLangGraphWorkflow:
    """Test suite for the LangGraphWorkflow class."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        with patch('langgraph_demo.workflow.ChatOpenAI') as mock_chat:
            mock_instance = MagicMock()
            mock_instance.invoke.return_value = MagicMock(content="Test response")
            mock_chat.return_value = mock_instance
            yield mock_instance

    def test_process_message(self, mock_llm):
        """Test processing a message through the workflow."""
        # Arrange
        workflow = LangGraphWorkflow()
        test_message = "Hello, world!"

        # Act
        response = workflow.process_message(test_message)

        # Assert
        assert response == "Test response"
        mock_llm.invoke.assert_called_once()
        
        # Check that the LLM was called with the correct messages
        args, _ = mock_llm.invoke.call_args
        messages = args[0]
        assert len(messages) == 2  # System message + user message
        assert messages[0].type == "system"
        assert messages[1].content == test_message

    def test_workflow_initialization(self, mock_llm):
        """Test that the workflow is properly initialized."""
        # Act
        workflow = LangGraphWorkflow()

        # Assert
        assert workflow is not None
        assert hasattr(workflow, 'workflow')
        assert hasattr(workflow, 'settings')
        assert hasattr(workflow, 'llm')

    def test_workflow_with_empty_message(self, mock_llm):
        """Test that the workflow handles empty messages gracefully."""
        # Arrange
        workflow = LangGraphWorkflow()
        empty_message = ""

        # Act & Assert
        with pytest.raises(ValueError):
            workflow.process_message(empty_message)
