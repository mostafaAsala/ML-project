# Sequence-to-Sequence Summarizer

This module provides functionality to train a sequence-to-sequence model for movie plot summarization using data generated from the LLMSummarizer. It leverages the Hugging Face transformers library to fine-tune pre-trained models for the summarization task.

## Overview

The `Seq2SeqSummarizerTrainer` class enables you to:

1. **Load LLM-generated summaries**: Use summaries generated by the `LLMSummarizer` as training data
2. **Prepare data for training**: Process and split the data into training, validation, and test sets
3. **Train a sequence-to-sequence model**: Fine-tune pre-trained models like T5 or BART for summarization
4. **Evaluate model performance**: Assess the quality of generated summaries
5. **Generate new summaries**: Use the trained model to summarize new movie plots

## Why Train a Seq2Seq Model?

While LLMs like GPT-4, Claude, or Mistral can generate high-quality summaries, they have several limitations:

- **Cost**: API calls to commercial LLMs can be expensive for large-scale summarization
- **Latency**: API calls introduce latency that may not be acceptable for real-time applications
- **Dependency**: Relying on external APIs introduces dependencies on third-party services
- **Customization**: Limited ability to customize the model for specific summarization styles

By training a sequence-to-sequence model on LLM-generated summaries, you get:

- **Cost-effective inference**: No API costs for generating summaries after training
- **Lower latency**: Local inference is typically faster than API calls
- **Independence**: No reliance on external services
- **Customization**: The model learns the specific summarization style of the LLM

## Installation

The module requires the following dependencies:

```bash
pip install torch transformers datasets pandas numpy scikit-learn tqdm
```

## Usage

### Basic Usage

```python
from seq2seq_summarizer_trainer import Seq2SeqSummarizerTrainer

# Create a trainer instance
trainer = Seq2SeqSummarizerTrainer(
    model_name="t5-small",  # Pre-trained model to use
    max_input_length=512,   # Maximum input length
    max_output_length=128   # Maximum output length
)

# Load summaries generated by LLMSummarizer
summaries = trainer.load_summaries_from_file("generated_summaries/summaries_example.json")

# Prepare data for training
train_dataset, val_dataset, test_dataset = trainer.prepare_data_from_summaries(
    summaries,
    test_size=0.1,  # 10% for testing
    val_size=0.1    # 10% for validation
)

# Train the model
trainer.train(
    output_dir="seq2seq_summarizer_model",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8
)

# Evaluate the model
metrics = trainer.evaluate()
print(f"Evaluation metrics: {metrics}")

# Generate a summary for a new movie plot
plot = "A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers."
summary = trainer.generate_summary(plot)
print(f"Generated summary: {summary}")
```

### Loading a Trained Model

```python
from seq2seq_summarizer_trainer import Seq2SeqSummarizerTrainer

# Load a trained model
trainer = Seq2SeqSummarizerTrainer.load_model("seq2seq_summarizer_model")

# Generate a summary
plot = "A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers."
summary = trainer.generate_summary(plot)
print(f"Generated summary: {summary}")
```

## Example Script

The `seq2seq_summarizer_example.py` script provides a complete example of how to use the `Seq2SeqSummarizerTrainer` class:

```bash
# Generate summaries using LLMSummarizer
python seq2seq_summarizer_example.py --generate --data movie_data.csv --sample-size 100

# Train a model using generated summaries
python seq2seq_summarizer_example.py --train --summaries generated_summaries/summaries_example.json --model t5-small --epochs 3

# Compare summaries generated by the trained model
python seq2seq_summarizer_example.py --compare --data movie_data.csv

# Do everything in one command
python seq2seq_summarizer_example.py --generate --train --compare --data movie_data.csv --sample-size 100
```

## Supported Models

The `Seq2SeqSummarizerTrainer` supports any sequence-to-sequence model from Hugging Face's model hub, including:

- **T5 models**: `t5-small`, `t5-base`, `t5-large`, etc.
- **BART models**: `facebook/bart-base`, `facebook/bart-large-cnn`, etc.
- **Pegasus models**: `google/pegasus-xsum`, `google/pegasus-cnn_dailymail`, etc.
- **mT5 models**: `google/mt5-small`, `google/mt5-base`, etc. (for multilingual summarization)

## Training Parameters

The `train` method accepts various parameters to customize the training process:

- `output_dir`: Directory to save the model
- `num_train_epochs`: Number of training epochs
- `per_device_train_batch_size`: Batch size for training
- `per_device_eval_batch_size`: Batch size for evaluation
- `warmup_steps`: Number of warmup steps
- `weight_decay`: Weight decay for regularization
- `logging_dir`: Directory for logs
- `logging_steps`: Number of steps between logging
- `evaluation_strategy`: When to evaluate ('no', 'steps', 'epoch')
- `save_strategy`: When to save checkpoints ('no', 'steps', 'epoch')
- `load_best_model_at_end`: Whether to load the best model at the end of training
- `metric_for_best_model`: Metric to use for best model selection
- `greater_is_better`: Whether higher is better for the metric

## Performance Considerations

- **Model size**: Larger models generally produce better summaries but require more memory and compute
- **Training data**: More training examples typically lead to better performance
- **Training time**: Training time increases with model size and dataset size
- **GPU memory**: Training larger models requires more GPU memory

## Integration with LLMSummarizer

The `Seq2SeqSummarizerTrainer` is designed to work seamlessly with the `LLMSummarizer` class:

1. Use `LLMSummarizer` to generate high-quality summaries from various LLM providers
2. Save the generated summaries to a JSON file
3. Use `Seq2SeqSummarizerTrainer` to train a model on these summaries
4. Deploy the trained model for efficient, cost-effective summarization

## Future Improvements

Potential enhancements for future versions:

1. **ROUGE metrics**: Add ROUGE score calculation for better evaluation
2. **Beam search parameters**: Allow customization of beam search parameters for generation
3. **Model distillation**: Support for distilling larger models into smaller ones
4. **Quantization**: Support for quantizing models for faster inference
5. **Batch processing**: Support for batch processing of multiple texts
