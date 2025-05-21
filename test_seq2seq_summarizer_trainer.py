"""
Unit tests for the Seq2SeqSummarizerTrainer class.
"""

import os
import json
import unittest
import tempfile
import shutil
import pandas as pd
from unittest.mock import patch, MagicMock

from seq2seq_summarizer_trainer import Seq2SeqSummarizerTrainer


class TestSeq2SeqSummarizerTrainer(unittest.TestCase):
    """Test cases for the Seq2SeqSummarizerTrainer class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        # Create a temporary directory for test data
        cls.test_dir = tempfile.mkdtemp()
        
        # Create test summaries
        cls.test_summaries = {
            "Movie 1": {
                "original_plot": "This is a long plot about a movie with various characters and events that happen throughout the story.",
                "summary": "A concise summary of the movie plot.",
                "timestamp": "2023-01-01T12:00:00"
            },
            "Movie 2": {
                "original_plot": "Another movie plot with different characters and a completely different storyline that unfolds over time.",
                "summary": "Another summary that captures the essence of the movie.",
                "timestamp": "2023-01-01T12:01:00"
            },
            "Movie 3": {
                "original_plot": "A third movie plot with its own unique characters and storyline that takes place in a different setting.",
                "summary": "A third summary that describes the movie briefly.",
                "timestamp": "2023-01-01T12:02:00"
            }
        }
        
        # Save test summaries to a file
        cls.summaries_path = os.path.join(cls.test_dir, "test_summaries.json")
        with open(cls.summaries_path, 'w', encoding='utf-8') as f:
            json.dump(cls.test_summaries, f)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test data."""
        shutil.rmtree(cls.test_dir)
    
    def test_init(self):
        """Test initialization of Seq2SeqSummarizerTrainer."""
        trainer = Seq2SeqSummarizerTrainer(
            model_name="t5-small",
            max_input_length=256,
            max_output_length=64,
            device="cpu"
        )
        
        self.assertEqual(trainer.model_name, "t5-small")
        self.assertEqual(trainer.max_input_length, 256)
        self.assertEqual(trainer.max_output_length, 64)
        self.assertEqual(trainer.device, "cpu")
    
    def test_load_summaries_from_file(self):
        """Test loading summaries from a file."""
        trainer = Seq2SeqSummarizerTrainer(device="cpu")
        
        # Test loading valid file
        summaries = trainer.load_summaries_from_file(self.summaries_path)
        self.assertEqual(len(summaries), 3)
        self.assertIn("Movie 1", summaries)
        self.assertIn("Movie 2", summaries)
        self.assertIn("Movie 3", summaries)
        
        # Test loading non-existent file
        with self.assertRaises(FileNotFoundError):
            trainer.load_summaries_from_file("non_existent_file.json")
    
    @patch('seq2seq_summarizer_trainer.Dataset')
    @patch('seq2seq_summarizer_trainer.train_test_split')
    def test_prepare_data_from_summaries(self, mock_train_test_split, mock_dataset):
        """Test preparing data from summaries."""
        # Set up mocks
        mock_train_df = pd.DataFrame([{"title": "Movie 1", "plot": "Plot 1", "summary": "Summary 1"}])
        mock_val_df = pd.DataFrame([{"title": "Movie 2", "plot": "Plot 2", "summary": "Summary 2"}])
        mock_test_df = pd.DataFrame([{"title": "Movie 3", "plot": "Plot 3", "summary": "Summary 3"}])
        
        mock_train_test_split.side_effect = [
            (mock_train_df, mock_test_df),
            (mock_train_df, mock_val_df)
        ]
        
        mock_train_dataset = MagicMock()
        mock_val_dataset = MagicMock()
        mock_test_dataset = MagicMock()
        
        mock_dataset.from_pandas.side_effect = [
            mock_train_dataset,
            mock_val_dataset,
            mock_test_dataset
        ]
        
        mock_train_dataset.map.return_value = mock_train_dataset
        mock_val_dataset.map.return_value = mock_val_dataset
        mock_test_dataset.map.return_value = mock_test_dataset
        
        # Create trainer
        trainer = Seq2SeqSummarizerTrainer(device="cpu")
        
        # Call the method
        train_dataset, val_dataset, test_dataset = trainer.prepare_data_from_summaries(
            self.test_summaries,
            test_size=0.2,
            val_size=0.1
        )
        
        # Verify results
        self.assertEqual(train_dataset, mock_train_dataset)
        self.assertEqual(val_dataset, mock_val_dataset)
        self.assertEqual(test_dataset, mock_test_dataset)
        
        # Verify the datasets were stored
        self.assertEqual(trainer.train_dataset, mock_train_dataset)
        self.assertEqual(trainer.eval_dataset, mock_val_dataset)
        self.assertEqual(trainer.test_dataset, mock_test_dataset)
    
    @patch('seq2seq_summarizer_trainer.Seq2SeqTrainer')
    @patch('seq2seq_summarizer_trainer.Seq2SeqTrainingArguments')
    @patch('seq2seq_summarizer_trainer.DataCollatorForSeq2Seq')
    def test_train(self, mock_data_collator, mock_training_args, mock_trainer):
        """Test training the model."""
        # Set up mocks
        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance
        
        # Create trainer
        trainer = Seq2SeqSummarizerTrainer(device="cpu")
        trainer.train_dataset = MagicMock()
        trainer.eval_dataset = MagicMock()
        
        # Call the method
        result = trainer.train(
            output_dir="test_output",
            num_train_epochs=2,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2
        )
        
        # Verify results
        self.assertEqual(result, mock_trainer_instance)
        self.assertEqual(trainer.trainer, mock_trainer_instance)
        
        # Verify the trainer was created with the correct arguments
        mock_trainer.assert_called_once()
        mock_training_args.assert_called_once()
        mock_data_collator.assert_called_once()
        
        # Verify the model was trained and saved
        mock_trainer_instance.train.assert_called_once()
        mock_trainer_instance.save_model.assert_called_once_with("test_output")
    
    def test_train_without_data(self):
        """Test training without preparing data first."""
        trainer = Seq2SeqSummarizerTrainer(device="cpu")
        
        with self.assertRaises(ValueError):
            trainer.train()
    
    @patch('seq2seq_summarizer_trainer.AutoTokenizer')
    @patch('seq2seq_summarizer_trainer.AutoModelForSeq2SeqLM')
    def test_load_model(self, mock_model, mock_tokenizer):
        """Test loading a model from disk."""
        # Set up mocks
        mock_tokenizer_instance = MagicMock()
        mock_model_instance = MagicMock()
        
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance
        
        # Call the method
        trainer = Seq2SeqSummarizerTrainer.load_model("test_model_path", device="cpu")
        
        # Verify results
        self.assertEqual(trainer.tokenizer, mock_tokenizer_instance)
        self.assertEqual(trainer.model, mock_model_instance)
        
        # Verify the model and tokenizer were loaded
        mock_tokenizer.from_pretrained.assert_called_once_with("test_model_path")
        mock_model.from_pretrained.assert_called_once_with("test_model_path")
    
    @patch('seq2seq_summarizer_trainer.Seq2SeqSummarizerTrainer.generate_summary')
    def test_generate_summary(self, mock_generate_summary):
        """Test generating a summary."""
        # Set up mock
        mock_generate_summary.return_value = "A generated summary."
        
        # Create trainer
        trainer = Seq2SeqSummarizerTrainer(device="cpu")
        
        # Call the method
        summary = trainer.generate_summary("This is a test plot.")
        
        # Verify results
        self.assertEqual(summary, "A generated summary.")
        mock_generate_summary.assert_called_once_with("This is a test plot.")


if __name__ == '__main__':
    unittest.main()
