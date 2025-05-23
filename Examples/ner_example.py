"""
Movie NER Model Example

This script demonstrates how to:
1. Train a movie NER model
2. Test the trained model
3. Compare before/after training results
4. Use the model for entity extraction
"""

import os
from ner_model import MovieNERModel, train_movie_ner_model


def demo_training():
    """Demonstrate training a movie NER model."""
    print("=" * 60)
    print("MOVIE NER MODEL TRAINING DEMO")
    print("=" * 60)
    
    # Check for movie data
    movie_data_files = [
        'wiki_movie_plots_deduped.csv',
        'wiki_movie_plots_deduped_cleaned.csv'
    ]
    
    movie_data_path = None
    for file_path in movie_data_files:
        if os.path.exists(file_path):
            movie_data_path = file_path
            print(f"Found movie data: {file_path}")
            break
    
    if not movie_data_path:
        print("No movie data found. Using fallback data for training.")
    
    print("\n1. Training NER model...")
    print("   This may take a few minutes...")
    
    # Train the model
    model_path = train_movie_ner_model(
        movie_data_path=movie_data_path,
        num_samples=500,  # Reasonable number for demo
        n_iter=50,        # Good balance of speed and performance
        model_save_path="saved_models/demo_ner_model"
    )
    
    print(f"\n✓ Model trained and saved to: {model_path}")
    return model_path


def demo_entity_extraction(model_path):
    """Demonstrate entity extraction with the trained model."""
    print("\n" + "=" * 60)
    print("ENTITY EXTRACTION DEMO")
    print("=" * 60)
    
    # Load the trained model
    model = MovieNERModel()
    model.load_model(model_path)
    
    # Test queries
    test_queries = [
        "I want action movies directed by Christopher Nolan",
        "Show me comedy films with Will Smith and Kevin Hart",
        "Find horror movies starring Lupita Nyong'o",
        "I love animated movies for family viewing",
        "Show me thriller films directed by Denis Villeneuve",
        "Find romantic comedies with Emma Stone",
        "I want sci-fi movies with Leonardo DiCaprio",
        "Show me drama films starring Meryl Streep",
        "Find western movies directed by Quentin Tarantino",
        "I love fantasy films with magical elements"
    ]
    
    print("Extracting entities from test queries:")
    print("-" * 60)
    
    for i, query in enumerate(test_queries, 1):
        entities = model.extract_entities(query)
        
        print(f"\n{i}. Query: {query}")
        
        found_any = False
        for entity_type, entity_list in entities.items():
            if entity_list:
                print(f"   {entity_type}: {entity_list}")
                found_any = True
        
        if not found_any:
            print("   No entities found")


def demo_before_after_comparison():
    """Show the difference between untrained and trained models."""
    print("\n" + "=" * 60)
    print("BEFORE/AFTER TRAINING COMPARISON")
    print("=" * 60)
    
    test_query = "I want action movies directed by Christopher Nolan"
    
    # Test with untrained model
    print("1. Testing with UNTRAINED model:")
    untrained_model = MovieNERModel()
    untrained_entities = untrained_model.extract_entities(test_query)
    print(f"   Query: {test_query}")
    print(f"   Entities: {untrained_entities}")
    
    # Train a small model quickly
    print("\n2. Training a model (quick training)...")
    trained_model = MovieNERModel()
    trained_model.prepare_training_data(num_samples=200)
    trained_model.train(n_iter=15)
    
    # Test with trained model
    print("\n3. Testing with TRAINED model:")
    trained_entities = trained_model.extract_entities(test_query)
    print(f"   Query: {test_query}")
    print(f"   Entities: {trained_entities}")
    
    # Compare results
    print("\n4. Comparison:")
    print(f"   Untrained: {untrained_entities}")
    print(f"   Trained:   {trained_entities}")
    
    # Check improvement
    untrained_total = sum(len(entities) for entities in untrained_entities.values())
    trained_total = sum(len(entities) for entities in trained_entities.values())
    
    if trained_total > untrained_total:
        print("   ✓ Training improved entity extraction!")
    else:
        print("   ⚠ Training may need more data or iterations")


def demo_batch_processing():
    """Demonstrate processing multiple queries at once."""
    print("\n" + "=" * 60)
    print("BATCH PROCESSING DEMO")
    print("=" * 60)
    
    # Train a quick model
    print("Training model for batch processing demo...")
    model = MovieNERModel()
    model.prepare_training_data(num_samples=300)
    model.train(n_iter=20)
    
    # Batch of queries
    queries = [
        "Show me action movies",
        "I want films by Steven Spielberg", 
        "Find movies with Tom Hanks",
        "Comedy films please",
        "Horror movies directed by Jordan Peele"
    ]
    
    print(f"\nProcessing {len(queries)} queries:")
    print("-" * 40)
    
    for i, query in enumerate(queries, 1):
        entities = model.extract_entities(query)
        print(f"\n{i}. {query}")
        
        for entity_type, entity_list in entities.items():
            if entity_list:
                print(f"   {entity_type}: {entity_list}")


def demo_custom_training_data():
    """Demonstrate training with custom data."""
    print("\n" + "=" * 60)
    print("CUSTOM TRAINING DATA DEMO")
    print("=" * 60)
    
    # Create custom training data
    custom_data = [
        ("I want movies directed by James Cameron", {"entities": [(27, 40, "DIRECTOR")]}),
        ("Show me films with Leonardo DiCaprio", {"entities": [(20, 36, "CAST")]}),
        ("Find action movies", {"entities": [(5, 11, "GENRE")]}),
        ("I love sci-fi films by Ridley Scott", {"entities": [(7, 13, "GENRE"), (23, 35, "DIRECTOR")]}),
        ("Comedy movies with Will Smith", {"entities": [(0, 6, "GENRE"), (19, 29, "CAST")]}),
    ]
    
    print("Custom training data:")
    for text, annotations in custom_data:
        print(f"  Text: {text}")
        for start, end, label in annotations['entities']:
            entity_text = text[start:end]
            print(f"    {label}: '{entity_text}'")
    
    # Train model with custom data
    print("\nTraining model with custom data...")
    model = MovieNERModel()
    model.training_data = custom_data[:4]  # Use most for training
    model.validation_data = custom_data[4:]  # Use last for validation
    
    try:
        model.train(n_iter=10)
        
        # Test the custom model
        print("\nTesting custom trained model:")
        test_query = "I want action films directed by James Cameron"
        entities = model.extract_entities(test_query)
        print(f"Query: {test_query}")
        print(f"Entities: {entities}")
        
    except Exception as e:
        print(f"Custom training failed: {e}")


def main():
    """Main demo function."""
    print("Movie NER Model - Complete Demo")
    print("This demo shows training and using a movie entity extraction model")
    print()
    
    try:
        # Demo 1: Full training
        model_path = demo_training()
        
        # Demo 2: Entity extraction
        demo_entity_extraction(model_path)
        
        # Demo 3: Before/after comparison
        demo_before_after_comparison()
        
        # Demo 4: Batch processing
        demo_batch_processing()
        
        # Demo 5: Custom training data
        demo_custom_training_data()
        
        print("\n" + "=" * 60)
        print("🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nKey takeaways:")
        print("- NER models need training to extract entities effectively")
        print("- More training data and iterations improve performance")
        print("- The model can extract multiple entity types from one query")
        print("- Custom training data can be used for specific use cases")
        print("- Trained models can be saved and loaded for reuse")
        
        print(f"\nTrained model saved at: {model_path}")
        print("You can load this model later using:")
        print("  model = MovieNERModel()")
        print(f"  model.load_model('{model_path}')")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
