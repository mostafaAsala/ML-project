"""
Unit tests for the LLMSummarizer class.
"""

import unittest
import os
import json
import tempfile
import pandas as pd
from unittest.mock import patch, MagicMock
from llm_summarizer import LLMSummarizer

class TestLLMSummarizer(unittest.TestCase):
    """Test cases for the LLMSummarizer class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a test dataframe
        self.test_data = pd.DataFrame({
            'Title': ['Test Movie 1', 'Test Movie 2'],
            'Plot': [
                'This is a test plot for movie 1. It has a beginning, middle, and end.',
                'This is a test plot for movie 2. Something happens and then something else happens.'
            ]
        })

        # Create a temporary directory for saving summaries
        self.temp_dir = tempfile.mkdtemp()

        # Create a summarizer with mock provider
        self.summarizer = LLMSummarizer(
            model_name='test-model',
            provider='openai',
            save_dir=self.temp_dir
        )

        # Set the test data
        self.summarizer.df = self.test_data
        self.summarizer.plot_col = 'Plot'
        self.summarizer.title_col = 'Title'

    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up temporary directory
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)

    def test_init(self):
        """Test initialization of LLMSummarizer."""
        summarizer = LLMSummarizer(
            model_name='gpt-3.5-turbo',
            provider='openai',
            max_tokens=100,
            temperature=0.5,
            non_spoiler=True
        )

        self.assertEqual(summarizer.model_name, 'gpt-3.5-turbo')
        self.assertEqual(summarizer.provider, 'openai')
        self.assertEqual(summarizer.max_tokens, 100)
        self.assertEqual(summarizer.temperature, 0.5)
        self.assertTrue(summarizer.non_spoiler)

    def test_validate_provider_and_model(self):
        """Test validation of provider and model."""
        # Valid provider and model
        summarizer = LLMSummarizer(model_name='gpt-3.5-turbo', provider='openai')
        self.assertEqual(summarizer.provider, 'openai')

        # Invalid provider
        with self.assertRaises(ValueError):
            LLMSummarizer(model_name='gpt-3.5-turbo', provider='invalid-provider')

        # Invalid model for provider
        with self.assertRaises(ValueError):
            LLMSummarizer(model_name='invalid-model', provider='openai')

        # Hugging Face accepts any model name
        summarizer = LLMSummarizer(model_name='any-model', provider='huggingface')
        self.assertEqual(summarizer.model_name, 'any-model')

    def test_load_data(self):
        """Test loading data from a file."""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
            self.test_data.to_csv(temp_file.name, index=False)
            temp_file_name = temp_file.name

        try:
            # Load the data
            summarizer = LLMSummarizer(model_name='gpt-3.5-turbo', provider='openai')
            summarizer.load_data(
                file_path=temp_file_name,
                plot_col='Plot',
                title_col='Title'
            )

            # Check that the data was loaded correctly
            self.assertEqual(len(summarizer.df), 2)
            self.assertEqual(summarizer.plot_col, 'Plot')
            self.assertEqual(summarizer.title_col, 'Title')

            # Test with sample_size
            summarizer.load_data(
                file_path=temp_file_name,
                plot_col='Plot',
                title_col='Title',
                sample_size=1
            )
            self.assertEqual(len(summarizer.df), 1)
        finally:
            # Clean up
            os.unlink(temp_file_name)

    @patch('llm_summarizer.LLMSummarizer._call_llm_api')
    def test_generate_summaries(self, mock_call_llm_api):
        """Test generating summaries."""
        # Mock the LLM API call
        mock_call_llm_api.return_value = "This is a test summary."

        # Generate summaries
        summaries = self.summarizer.generate_summaries(batch_size=1, delay=0)

        # Check that summaries were generated
        self.assertEqual(len(summaries), 2)
        self.assertIn('Test Movie 1', summaries)
        self.assertIn('Test Movie 2', summaries)
        self.assertEqual(summaries['Test Movie 1']['summary'], "This is a test summary.")

        # Check that the API was called with the correct prompt
        self.assertEqual(mock_call_llm_api.call_count, 2)

    def test_save_and_load_summaries(self):
        """Test saving and loading summaries."""
        # Create some test summaries
        self.summarizer.summaries = {
            'Test Movie 1': {
                'original_plot': 'This is a test plot.',
                'summary': 'This is a test summary.',
                'timestamp': '2023-01-01T00:00:00'
            }
        }

        # Save the summaries
        file_path = self.summarizer.save_summaries(file_name='test_summaries.json')

        # Check that the file was created
        self.assertTrue(os.path.exists(file_path))

        # Create a new summarizer
        new_summarizer = LLMSummarizer(
            model_name='gpt-3.5-turbo',
            provider='openai'
        )

        # Load the summaries
        new_summarizer.load_summaries(file_path)

        # Check that the summaries were loaded correctly
        self.assertEqual(len(new_summarizer.summaries), 1)
        self.assertIn('Test Movie 1', new_summarizer.summaries)
        self.assertEqual(
            new_summarizer.summaries['Test Movie 1']['summary'],
            'This is a test summary.'
        )

    @patch('openai.ChatCompletion.create')
    def test_call_openai_api(self, mock_create):
        """Test calling the OpenAI API."""
        # Skip if openai package is not installed
        try:
            import openai
        except ImportError:
            self.skipTest("openai package not installed")

        # Mock the OpenAI API response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "This is a test summary."
        mock_create.return_value = mock_response

        # Call the API
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            summarizer = LLMSummarizer(model_name='gpt-3.5-turbo', provider='openai')
            result = summarizer._call_openai_api("Test prompt")

        # Check the result
        self.assertEqual(result, "This is a test summary.")

        # Check that the API was called with the correct parameters
        mock_create.assert_called_once()
        args, kwargs = mock_create.call_args
        self.assertEqual(kwargs['model'], 'gpt-3.5-turbo')
        self.assertEqual(kwargs['messages'][1]['content'], "Test prompt")

    @patch('mistralai.client.MistralClient.chat')
    def test_call_mistral_api(self, mock_chat):
        """Test calling the Mistral AI API."""
        # Skip if mistralai package is not installed
        try:
            import mistralai
        except ImportError:
            self.skipTest("mistralai package not installed")

        # Mock the Mistral API response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "This is a test summary from Mistral."
        mock_chat.return_value = mock_response

        # Call the API
        with patch.dict('os.environ', {'MISTRAL_API_KEY': 'test-key'}):
            summarizer = LLMSummarizer(model_name='mistral-small', provider='mistral')
            result = summarizer._call_mistral_api("Test prompt")

        # Check the result
        self.assertEqual(result, "This is a test summary from Mistral.")

        # Check that the API was called with the correct parameters
        mock_chat.assert_called_once()
        args, kwargs = mock_chat.call_args
        self.assertEqual(kwargs['model'], 'mistral-small')
        self.assertEqual(kwargs['messages'][1].content, "Test prompt")

if __name__ == '__main__':
    unittest.main()
