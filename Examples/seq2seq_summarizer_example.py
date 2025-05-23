"""
Example script demonstrating how to use the Seq2SeqSummarizerTrainer class.

This script shows:
1. How to generate summaries using LLMSummarizer
2. How to train a sequence-to-sequence model using those summaries
3. How to evaluate the trained model
4. How to use the model for generating new summaries
"""

import os
import argparse
import pandas as pd
from llm_summarizer import LLMSummarizer
from seq2seq_summarizer_trainer import Seq2SeqSummarizerTrainer

def generate_llm_summaries(
    data_path: str,
    provider: str = "mistral",
    model_name: str = "mistral-small",
    sample_size: int = 100,
    batch_size: int = 10,
    delay: float = 1.0
):
    """
    Generate summaries using LLMSummarizer.
    
    Parameters:
    -----------
    data_path : str
        Path to the movie data CSV file
    provider : str, default='mistral'
        LLM provider to use
    model_name : str, default='mistral-small'
        Model name to use
    sample_size : int, default=100
        Number of movies to sample
    batch_size : int, default=10
        Batch size for generating summaries
    delay : float, default=1.0
        Delay between API calls
        
    Returns:
    --------
    str : Path to the saved summaries file
    """
    print(f"Generating summaries using {provider} model: {model_name}")
    
    # Create a summarizer instance
    summarizer = LLMSummarizer(
        model_name=model_name,
        provider=provider,
        max_tokens=150,
        non_spoiler=True
    )
    
    # Load data
    print(f"Loading data from {data_path}")
    summarizer.load_data(
        file_path=data_path,
        plot_col="Plot",
        title_col="Title",
        sample_size=sample_size
    )
    
    # Generate summaries
    print(f"Generating summaries for {sample_size} movies")
    summaries = summarizer.generate_summaries(batch_size=batch_size, delay=delay)
    
    # Save summaries
    file_path = summarizer.save_summaries()
    print(f"Summaries saved to {file_path}")
    
    return file_path

def train_seq2seq_model(
    summaries_path: str,
    model_name: str = "t5-small",
    output_dir: str = "seq2seq_summarizer_model",
    num_epochs: int = 3,
    batch_size: int = 4
):
    """
    Train a sequence-to-sequence model using generated summaries.
    
    Parameters:
    -----------
    summaries_path : str
        Path to the JSON file containing summaries
    model_name : str, default='t5-small'
        Pre-trained model to use
    output_dir : str, default='seq2seq_summarizer_model'
        Directory to save the trained model
    num_epochs : int, default=3
        Number of training epochs
    batch_size : int, default=4
        Batch size for training
        
    Returns:
    --------
    Seq2SeqSummarizerTrainer : Trained model
    """
    print(f"Training sequence-to-sequence model using {model_name}")
    
    # Create a trainer instance
    trainer = Seq2SeqSummarizerTrainer(
        model_name=model_name,
        max_input_length=512,
        max_output_length=128
    )
    
    # Load summaries
    print(f"Loading summaries from {summaries_path}")
    summaries = trainer.load_summaries_from_file(summaries_path)
    
    # Prepare data
    print("Preparing data for training")
    train_dataset, val_dataset, test_dataset = trainer.prepare_data_from_summaries(
        summaries,
        test_size=0.1,
        val_size=0.1
    )
    
    # Train the model
    print(f"Training model for {num_epochs} epochs")
    trainer.train(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size
    )
    
    # Evaluate the model
    print("Evaluating model")
    metrics = trainer.evaluate()
    print(f"Evaluation metrics: {metrics}")
    
    return trainer

def compare_summaries(
    model_path: str,
    data_path: str,
    sample_size: int = 5
):
    """
    Compare summaries generated by the trained model with the original plots.
    
    Parameters:
    -----------
    model_path : str
        Path to the trained model
    data_path : str
        Path to the movie data CSV file
    sample_size : int, default=5
        Number of movies to sample
    """
    print(f"Comparing summaries using model from {model_path}")
    
    # Load the trained model
    trainer = Seq2SeqSummarizerTrainer.load_model(model_path)
    
    # Load some movie data
    df = pd.read_csv(data_path)
    sample_df = df.sample(n=sample_size, random_state=42)
    
    # Generate summaries
    print("Generating summaries")
    for _, row in sample_df.iterrows():
        title = row["Title"]
        plot = row["Plot"]
        
        # Skip if plot is too short
        if len(str(plot).split()) < 20:
            continue
        
        # Generate summary
        summary = trainer.generate_summary(plot)
        
        # Print results
        print(f"\nTitle: {title}")
        print(f"Plot: {plot[:200]}...")
        print(f"Generated summary: {summary}")

def main():
    print("222222222222222222222222")
    """Main function to run the example script."""
    parser = argparse.ArgumentParser(description="Seq2Seq Summarizer Example")
    parser.add_argument("--data", default="wiki_movie_plots_deduped_cleaned.csv", help="Path to movie data CSV file")
    parser.add_argument("--generate", action="store_true", help="Generate summaries using LLMSummarizer")
    parser.add_argument("--train", action="store_true", help="Train a sequence-to-sequence model")
    parser.add_argument("--compare", action="store_true", help="Compare summaries")
    parser.add_argument("--summaries",default="generated_summaries\summaries_mistral_mistral_small_20250522_000419.json", help="Path to summaries JSON file (for training)")
    parser.add_argument("--model", default="t5-small", help="Pre-trained model to use")
    parser.add_argument("--output", default="seq2seq_summarizer_model", help="Directory to save the trained model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of movies to sample")
    parser.add_argument("--provider", default="mistral", help="LLM provider to use")
    parser.add_argument("--llm-model", default="mistral-small", help="LLM model to use")
    
    
    args = parser.parse_args()
    # If no arguments provided, show help
    if not (args.generate or args.train or args.compare):
        parser.print_help()
        return
    print("args ------------------------------",args)

    # Generate summaries
    if args.generate:
        summaries_path = generate_llm_summaries(
            data_path=args.data,
            provider=args.provider,
            model_name=args.llm_model,
            sample_size=args.sample_size,
            batch_size=10,
            delay=1.0
        )
        
        # If training is also requested, use the generated summaries
        if args.train and not args.summaries:
            args.summaries = summaries_path
    
    # Train model
    if args.train:
        if not args.summaries:
            print("Error: No summaries file provided. Use --summaries or --generate.")
            return
        
        trainer = train_seq2seq_model(
            summaries_path=args.summaries,
            model_name=args.model,
            output_dir=args.output,
            num_epochs=args.epochs,
            batch_size=args.batch_size
        )
    
    # Compare summaries
    if args.compare:
        if not os.path.exists(args.output):
            print(f"Error: Model directory not found: {args.output}")
            return
        
        compare_summaries(
            model_path=args.output,
            data_path=args.data,
            sample_size=5
        )

if __name__ == "__main__":
    main()
