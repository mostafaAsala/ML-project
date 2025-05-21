"""
Example script demonstrating how to use the LLMSummarizer class to generate non-spoiler summaries.

This script shows:
1. How to create an LLMSummarizer instance
2. How to load movie data
3. How to generate non-spoiler summaries
4. How to save and load summaries
"""

import os
import pandas as pd
import json
from llm_summarizer import LLMSummarizer

def generate_summaries_example():
    """
    Example of generating non-spoiler summaries using LLMSummarizer.
    """
    print("=== Generating Non-Spoiler Summaries ===")

    # Check if API key is set
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("Set it with: export OPENAI_API_KEY='your-api-key'")
        print("For this example, we'll use a mock implementation.")

    # Create a summarizer instance
    summarizer = LLMSummarizer(
        model_name="mistral-small'",
        provider="mistral",
        max_tokens=150,
        temperature=0.7,
        non_spoiler=True
    )

    # Load data
    print("\nLoading movie data...")
    summarizer.load_data(
        file_path="wiki_movie_plots_deduped_cleaned.csv",
        plot_col="Plot",
        title_col="Title",
        sample_size=5  # Small sample for testing
    )

    # Generate summaries
    print("\nGenerating summaries...")
    summaries = summarizer.generate_summaries(batch_size=5, delay=1.0)

    # Print sample summaries
    print("\nGenerated Summaries:")
    for title, data in summaries.items():
        print(f"\nTitle: {title}")
        print(f"Summary: {data['summary'][:100]}...")

    # Save summaries
    print("\nSaving summaries...")
    file_path = summarizer.save_summaries()
    print(f"Summaries saved to: {file_path}")

    return file_path

"""def load_summaries_example(file_path):
    ""
    Example of loading previously generated summaries.

    Parameters:
    -----------
    file_path : str
        Path to the saved summaries file
    ""
    print("\n=== Loading Saved Summaries ===")

    # Create a new summarizer instance
    summarizer = LLMSummarizer(
        model_name="mistral-small",
        provider="mistral"
    )

    # Load summaries
    summarizer.load_summaries(file_path)

    # Print the loaded summaries
    print(f"\nLoaded {len(summarizer.summaries)} summaries:")
    for title, data in list(summarizer.summaries.items())[:3]:  # Show first 3
        print(f"\nTitle: {title}")
        print(f"Summary: {data['summary'][:100]}...")

def compare_models_example():
    ""
    Example of comparing summaries from different LLM models.
    ""
    print("\n=== Comparing Different LLM Models ===")

    # Check if API keys are set
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    mistral_key = os.environ.get("MISTRAL_API_KEY")

    if not openai_key and not anthropic_key and not mistral_key:
        print("Warning: No API keys found. This example requires at least one API key.")
        print("Set them with: export OPENAI_API_KEY='your-key', export ANTHROPIC_API_KEY='your-key', or export MISTRAL_API_KEY='your-key'")
        return

    # List of models to compare
    models_to_compare = []

    if openai_key:
        models_to_compare.append(("openai", "gpt-3.5-turbo"))

    if anthropic_key:
        models_to_compare.append(("anthropic", "claude-instant-1"))

    if mistral_key:
        models_to_compare.append(("mistral", "mistral-small"))

    if not models_to_compare:
        print("No valid models to compare. Please set at least one API key.")
        return

    # Sample movie for comparison
    sample_data = {
        "Title": "The Shawshank Redemption",
        "Plot": "Andy Dufresne is sentenced to two consecutive life terms in prison for the murders of his wife and her lover and is sentenced to a tough prison. However, only Andy knows he didn't commit the crimes. While there, he forms a friendship with Red, experiences brutality of prison life, adapts, helps the warden, etc., all in 19 years."
    }

    # Create a small DataFrame
    df = pd.DataFrame([sample_data])

    # Compare summaries from different models
    all_summaries = {}

    for provider, model_name in models_to_compare:
        print(f"\nGenerating summary using {provider} model: {model_name}")

        # Create summarizer
        summarizer = LLMSummarizer(
            model_name=model_name,
            provider=provider,
            max_tokens=150,
            non_spoiler=True
        )

        # Set the data
        summarizer.df = df
        summarizer.plot_col = "Plot"
        summarizer.title_col = "Title"

        # Generate summary
        summaries = summarizer.generate_summaries(batch_size=1, delay=0)

        # Store the summary
        if summaries and "The Shawshank Redemption" in summaries:
            all_summaries[f"{provider}_{model_name}"] = summaries["The Shawshank Redemption"]["summary"]

    # Print comparison
    print("\nSummary Comparison:")
    for model, summary in all_summaries.items():
        print(f"\n{model}:")
        print(summary)"""

def mistral_example():
    """
    Example of using Mistral AI for generating summaries.
    """
    print("\n=== Using Mistral AI for Summaries ===")

    # Check if Mistral API key is set
    mistral_key = os.environ.get("MISTRAL_API_KEY")
    if not mistral_key:
        print("Warning: MISTRAL_API_KEY environment variable not set.")
        print("Set it with: export MISTRAL_API_KEY='your-api-key'")
        print("Skipping Mistral example.")
        return

    # Create a summarizer instance with Mistral
    summarizer = LLMSummarizer(
        model_name="mistral-small",
        provider="mistral",
        max_tokens=150,
        temperature=0.7,
        non_spoiler=True
    )

    # Create a small sample dataset
    sample_data = {
        "Title": ["Inception", "The Matrix"],
        "Plot": [
            "Dom Cobb is a skilled thief, the absolute best in the dangerous art of extraction, stealing valuable secrets from deep within the subconscious during the dream state, when the mind is at its most vulnerable.",
            "A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers."
        ]
    }

    # Create a DataFrame
    df = pd.DataFrame(sample_data)

    # Set the data
    summarizer.df = df
    summarizer.plot_col = "Plot"
    summarizer.title_col = "Title"

    # Generate summaries
    print("\nGenerating summaries with Mistral AI...")
    try:
        summaries = summarizer.generate_summaries(batch_size=2, delay=1.0)

        # Print the summaries
        print("\nGenerated Summaries:")
        for title, data in summaries.items():
            print(f"\nTitle: {title}")
            print(f"Summary: {data['summary']}")
    except Exception as e:
        print(f"Error using Mistral AI: {str(e)}")
        print("Make sure you have installed the Mistral AI package: pip install mistralai")

def main():
    """Main function to run all examples."""
    # Generate summaries
    file_path = generate_summaries_example()

    """# Load saved summaries
    if file_path and os.path.exists(file_path):
        load_summaries_example(file_path)
    
    # Compare different models
    compare_models_example()"""

    # Mistral AI example
    mistral_example()

if __name__ == "__main__":
    main()
