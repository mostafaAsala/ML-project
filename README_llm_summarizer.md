# LLM Summarizer

A module for generating non-spoiler summaries of movie plots using Large Language Models (LLMs).

## Overview

The `LLMSummarizer` class provides functionality to:

- Generate concise, non-spoiler summaries of movie plots using various LLM providers
- Support for multiple LLM providers (OpenAI, Hugging Face, Anthropic, local models)
- Load and process movie data
- Save and load generated summaries

## Installation

No additional installation is required beyond the dependencies already used in the project. However, depending on which LLM provider you want to use, you may need to install additional packages:

- For OpenAI: `pip install openai`
- For Anthropic: `pip install anthropic`
- For Hugging Face: `pip install transformers`

## Usage

### Basic Usage

```python
from llm_summarizer import LLMSummarizer

# Create a summarizer instance
summarizer = LLMSummarizer(
    model_name="gpt-3.5-turbo",
    provider="openai",
    max_tokens=150,
    non_spoiler=True
)

# Load data
summarizer.load_data(
    file_path="movie_data.csv",
    plot_col="Plot",
    title_col="Title",
    sample_size=10  # Optional: limit to 10 movies
)

# Generate summaries
summaries = summarizer.generate_summaries()

# Save summaries
summarizer.save_summaries()
```

### Loading Saved Summaries

```python
# Load previously generated summaries
summarizer = LLMSummarizer(model_name="gpt-3.5-turbo", provider="openai")
summarizer.load_summaries("generated_summaries/summaries_openai_gpt-3.5-turbo_20230615_123456.json")
```

### Comparing Different Models

```python
# Create summarizers with different models
summarizer1 = LLMSummarizer(model_name="gpt-3.5-turbo", provider="openai")
summarizer2 = LLMSummarizer(model_name="claude-instant-1", provider="anthropic")

# Load the same data for both
summarizer1.load_data("movie_data.csv", sample_size=5)
summarizer2.load_data("movie_data.csv", sample_size=5)

# Generate summaries
summaries1 = summarizer1.generate_summaries()
summaries2 = summarizer2.generate_summaries()

# Compare results
for title in summaries1:
    if title in summaries2:
        print(f"Title: {title}")
        print(f"OpenAI: {summaries1[title]['summary'][:100]}...")
        print(f"Anthropic: {summaries2[title]['summary'][:100]}...")
        print()
```

## API Keys

The `LLMSummarizer` will look for API keys in environment variables:

- OpenAI: `OPENAI_API_KEY`
- Hugging Face: `HUGGINGFACE_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- Mistral AI: `MISTRAL_API_KEY`

You can also provide the API key directly when creating the summarizer:

```python
summarizer = LLMSummarizer(
    model_name="gpt-3.5-turbo",
    provider="openai",
    api_key="your-api-key-here"
)
```

## Supported Models

### OpenAI
- gpt-3.5-turbo
- gpt-4

### Anthropic
- claude-instant-1
- claude-2
- claude-3-opus
- claude-3-sonnet
- claude-3-haiku

### Mistral AI
- mistral-tiny
- mistral-small
- mistral-medium
- mistral-large

### Hugging Face
- Any model available on the Hugging Face Hub

### Local Models
- llama2
- mistral
- (Any locally hosted model)

## Parameters

### LLMSummarizer

- `model_name`: Name of the LLM model to use
- `provider`: LLM provider ('openai', 'huggingface', 'anthropic', 'local')
- `api_key`: API key for the LLM provider
- `save_dir`: Directory to save generated summaries
- `max_tokens`: Maximum number of tokens for generated summaries
- `temperature`: Temperature parameter for LLM generation (higher = more creative)
- `non_spoiler`: Whether to generate non-spoiler summaries

### load_data

- `file_path`: Path to the CSV file containing movie data
- `plot_col`: Column name containing the movie plot text
- `title_col`: Column name containing the movie title
- `sample_size`: Number of movies to sample
- `random_state`: Random seed for reproducibility when sampling

### generate_summaries

- `batch_size`: Number of summaries to generate in each batch
- `delay`: Delay between API calls in seconds to avoid rate limits

## Example

See `llm_summarizer_example.py` for a complete example of how to use the `LLMSummarizer` class.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
