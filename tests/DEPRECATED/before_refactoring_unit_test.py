import unittest
from unittest.mock import patch, MagicMock
import os
import openai
import json
from config import *
from learn import TaskIdentificationAgent, HumanLLM
from langchain_core.messages.ai import AIMessage
from langchain_core.messages import HumanMessage, SystemMessage

class TestCollabGPTCreator(unittest.TestCase):

    def setUp(self):
        # Mock env setup
        os.environ['user_id'] = 'test_before_refactor'
        # Setup any required objects or state before each test
        self.agent = MagicMock()
        self.special_criteria = {
            'some_key': 'some_value',
            'local_var': 'local_value'
        }
        self.available_locals = {
            'local_var': 'local_value'
        }
    
    @patch('learn.openai.ChatCompletion.create')
    @patch('learn.HumanLLMMonitor.get_learnt_tasks', return_value=["Mocked task 1", "Mocked task 2"])
    @patch('learn.HumanLLMMonitor.get_failed_tasks', return_value=["Mocked failed task 1", "Mocked failed task 2"])
    @patch('learn.HumanLLMMonitor.add_agent_data')
    @patch('learn.HumanLLMMonitor.initialize_class_db')
    @patch('learn.HumanLLMMonitor.CallHumanLLM')
    def test_inference_checks(self, mock_call_human_llm, mock_initialize_class_db, mock_add_agent_data, mock_get_failed_tasks, mock_get_learnt_tasks, mock_openai):
        # Mock the response from OpenAI's API
        mock_openai.return_value = {"choices": [{"message": {"content": "Mocked response"}}]}
        
        # Mock the CallHumanLLM method
        mock_call_human_llm.return_value = "Mocked task result"

        agent = TaskIdentificationAgent(default_llm_choice="default", envs=[], special_criteria={})
        result = agent.identify_best_task()
        self.assertIsNotNone(result)
        self.assertEqual(result, "Mocked task result")


    @patch('utils.llm_utils.smart_input', return_value='Z')  # Mock smart_input to always return 'Z' (continue)
    @patch('learn.HumanLLMMonitor.add_agent_data')
    @patch('learn.HumanLLMMonitor.load_prompt', return_value="Mocked prompt")
    @patch('learn.HumanLLMMonitor.pre_inference', return_value=([SystemMessage(content="System prompt"), HumanMessage(content="User message")], "", True, False, None, None, None))
    @patch('learn.HumanLLMMonitor.post_inference', return_value=("", "", 0))
    @patch('learn.HumanLLMMonitor.process_output', return_value=(AIMessage(content="Processed LLM output"), None, None))
    @patch('learn.HumanLLMMonitor._log_entry')
    def test_CallHumanLLM(self, mock_log, mock_llm_ouput, mockpost_inference, mockpre_inference, mock_load_prompt, mock_add_agent_data, mock_smart_input):
        # Create an instance of HumanLLMMonitor
        monitor = HumanLLM(agent_name="TestAgent")
        
        # Mock input messages
        original_input_messages = [
            SystemMessage(content="System prompt"),
            HumanMessage(content="User message")
        ]
        default_llm_function = lambda messages: AIMessage(content="Default LLM response")
        premium_llm_function = lambda messages: AIMessage(content="Premium LLM response")
        callable_system_message = None
        system_prompt_template = "System prompt template"
        user_message = "Test user message"
        return_message_content_only = True
        function_calling = False
        temperature_min = 0.1
        timeout_seconds = 300
        stream_output = False
        use_default_llm = True
        model_choice = None
        temperature_max = 0.7
        task_name = None
        prompt_directory = "prompts"
        
        # Call the function
        result = monitor.invoke(
            original_input_messages, default_llm_function, premium_llm_function,
            callable_system_message, system_prompt_template, user_message,
            return_message_content_only, function_calling, temperature_min, timeout_seconds,
            stream_output, use_default_llm, model_choice, temperature_max, task_name, prompt_directory
        )
        
        # Check if the result is a list
        self.assertIsInstance(result, list)
        
        # Additional checks can be added based on expected behavior of the function
        self.assertTrue(len(result) > 0)
        self.assertIsInstance(result[0], str)

    @patch('utils.llm_utils.smart_input', return_value='Z')  # Mock smart_input to always return 'Z' (continue)
    @patch('learn.HumanLLMMonitor.add_agent_data')
    @patch('learn.HumanLLMMonitor.pre_inference', return_value=([SystemMessage(content="System prompt"), HumanMessage(content="User message")], "", True, False, None, None, None))
    @patch('learn.HumanLLMMonitor.post_inference', return_value=("", "", 0))
    @patch('learn.HumanLLMMonitor.process_output', return_value=(AIMessage(content="Processed LLM output"), None, None))  # Mock process_output
    @patch('utils.llm_utils.WebsocketServer.send_message')  # Mock WebSocket send_message method
    @patch('learn.HumanLLMMonitor._log_entry')  # Mock WebSocket send_message method
    def test_CallHumanLLM_socket_mode(self, mock_log, mock_send_message, mock_process_output, mock_after_infer, mock_before_infer, mock_add_agent_data, mock_smart_input):
        # Enable WebSocket mode
        HumanLLM.use_websocket = True

        # Create an instance of HumanLLMMonitor
        monitor = HumanLLM(agent_name="TestAgent")
        
        # Mock input messages
        original_input_messages = [
            SystemMessage(content="System prompt"),
            HumanMessage(content="User message")
        ]
        default_llm_function = MagicMock(return_value=AIMessage(content="Default LLM response"))
        premium_llm_function = MagicMock(return_value=AIMessage(content="Premium LLM response"))
        callable_system_message = None
        system_prompt_template = "System prompt template"
        user_message = "Test user message"
        return_message_content_only = True
        function_calling = False
        temperature_min = 0.1
        timeout_seconds = 300
        stream_output = False
        use_default_llm = True
        model_choice = None
        temperature_max = 0.7
        task_name = None
        prompt_directory = "prompts"

        # Call the function
        result = monitor.invoke(
            original_input_messages, default_llm_function, premium_llm_function,
            callable_system_message, system_prompt_template, user_message,
            return_message_content_only, function_calling, temperature_min, timeout_seconds,
            stream_output, use_default_llm, model_choice, temperature_max, task_name, prompt_directory
        )

        # Check if the result is a list
        self.assertIsInstance(result, list)

        # Additional checks can be added based on expected behavior of the function
        self.assertTrue(len(result) > 0)
        self.assertIsInstance(result[0], str)
        self.assertEqual(result[0], "Processed LLM output")

        # Ensure WebSocket send_message was called
        mock_send_message.assert_called()

    @patch('utils.llm_utils.smart_input', return_value='Z')  # Mock smart_input to always return 'Z' (continue)
    @patch('learn.HumanLLMMonitor.pre_inference', return_value=([SystemMessage(content="System prompt"), HumanMessage(content="User message")], "", True, False, None, None, None))
    @patch('learn.HumanLLMMonitor.post_inference', return_value=("", "", 0))
    @patch.object(HumanLLM, 'process_output', return_value=(AIMessage(content="Processed LLM output"), None, None))  # Mock process_output
    @patch('utils.llm_utils.WebsocketServer.send_message')  # Mock WebSocket send_message method
    def test_CallHumanLLM_socket_mode(self, mock_send_message, mock_process_output, mock_after_infer, mock_before_infer, mock_smart_input):
        # Enable WebSocket mode
        HumanLLM.use_websocket = False

        # Create an instance of HumanLLMMonitor
        monitor = HumanLLM(agent_name="TestAgent")
        
        # Mock input messages
        original_input_messages = [
            SystemMessage(content="System prompt"),
            HumanMessage(content="User message")
        ]
        default_llm_function = MagicMock(return_value=AIMessage(content="Default LLM response"))
        premium_llm_function = MagicMock(return_value=AIMessage(content="Premium LLM response"))
        callable_system_message = None
        system_prompt_template = "System prompt template"
        user_message = "Test user message"
        return_message_content_only = True
        function_calling = False
        temperature_min = 0.1
        timeout_seconds = 300
        stream_output = False
        use_default_llm = True
        model_choice = None
        temperature_max = 0.7
        task_name = None
        prompt_directory = "prompts"

        # Call the function
        result = monitor.invoke(
            original_input_messages, default_llm_function, premium_llm_function,
            callable_system_message, system_prompt_template, user_message,
            return_message_content_only, function_calling, temperature_min, timeout_seconds,
            stream_output, use_default_llm, model_choice, temperature_max, task_name, prompt_directory
        )

        # Check if the result is a list
        self.assertIsInstance(result, list)

        # Additional checks can be added based on expected behavior of the function
        self.assertTrue(len(result) > 0)
        self.assertIsInstance(result[0], str)
        self.assertEqual(result[0], "Processed LLM output")

        # Ensure WebSocket send_message was called
        mock_send_message.assert_not_called()

    @patch('utils.llm_utils.UnifiedVectorDB')
    @patch('utils.llm_utils.HumanLLMMonitor._check_and_init_vector_db')
    def test_add_agent_data(self, mock_check_and_init_vector_db, MockUnifiedVectorDB):
        # Mock the UnifiedVectorDB instance
        mock_vectordb_instance = MockUnifiedVectorDB.return_value

        # Create an instance of HumanLLMMonitor
        monitor = HumanLLM(agent_name="TestAgent")

        # Ensure the database is initialized
        HumanLLM.configure_vector_store(embedding_function="text-embedding-ada-002", reset_db_indices=True)
        
        # Set the common_vectordb to the mock instance
        HumanLLM.common_vectordb = mock_vectordb_instance

        # Call the method to add agent data
        monitor.add_agent_data(
            agent_name="TestAgent",
            data_key="test_key",
            data_value="test_value",
            function_name="test_function",
            task_id=False,
            before_after="before",
            user_id="test_user",
            step_id=1,
            task_type="test_tache",
            score=0.9,
            metadata={"test_meta_key": "test_meta_value"}
        )

        # Check if add_texts was called with the correct arguments
        self.assertTrue(mock_vectordb_instance._add_texts.called)
        call_args = mock_vectordb_instance._add_texts.call_args
        self.assertEqual(len(call_args), 2)  # Two arguments: texts and metadatas
        texts = call_args[1]['texts']  # call_args is a tuple, the first element contains the arguments
        metadatas = call_args[1]['metadatas']
        self.assertEqual(len(texts), 1)
        self.assertEqual(len(metadatas), 1)
        metadata = metadatas[0]
        self.assertEqual(metadata["agent_name"], "TestAgent")
        self.assertEqual(metadata["data_key"], "test_key")
        self.assertEqual(metadata["function_name"], "test_function")
        self.assertEqual(metadata["before_after"], "before")
        self.assertEqual(metadata["user_id"], "test_user")
        self.assertEqual(metadata["step_id"], 1)
        self.assertEqual(metadata["task_type"], "test_tache")
        self.assertEqual(metadata["score"], 0.9)
        self.assertEqual(metadata["test_meta_key"], "test_meta_value")

    @patch('utils.llm_utils.UnifiedVectorDB')
    @patch('utils.llm_utils.HumanLLMMonitor._check_and_init_vector_db')
    def test_get_agent_data(self, mock_check_and_init_vector_db, MockUnifiedVectorDB):
        # Mock the UnifiedVectorDB instance
        mock_vectordb_instance = MockUnifiedVectorDB.return_value
        mock_vectordb_instance.query.return_value = [MagicMock(page_content='{"test_key":"test_value"}', metadata={"metadata_key": "metadata_value"})]

        # Create an instance of HumanLLMMonitor
        monitor = HumanLLM(agent_name="TestAgent")

        # Ensure the database is initialized
        HumanLLM.configure_vector_store(embedding_function="text-embedding-ada-002", reset_db_indices=True)
        
        # Set the common_vectordb to the mock instance
        HumanLLM.common_vectordb = mock_vectordb_instance
        
        # Call the method to get agent data
        data, results = monitor.get_agent_data(agent_name="TestAgent", data_key="test_key")
        
        # Check the return values
        self.assertEqual(data, [{"test_key": "test_value"}])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].metadata["metadata_key"], "metadata_value")


if __name__ == '__main__':
    unittest.main()