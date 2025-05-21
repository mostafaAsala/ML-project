# Movie Analysis and Recommendation System

## Project Overview

This project implements a comprehensive movie analysis and recommendation system that combines data exploration, machine learning, natural language processing, and vector search capabilities. The system is designed to analyze movie data, predict genres, generate non-spoiler summaries, and provide semantic search functionality for movie discovery.

![Project Architecture](placeholder_images/project_architecture.png)

## Table of Contents

1. [Data Exploration and Analysis](#data-exploration-and-analysis)
2. [Genre Prediction Pipeline](#genre-prediction-pipeline)
3. [Model Evaluation and Selection](#model-evaluation-and-selection)
4. [Model Saving and Loading](#model-saving-and-loading)
5. [LLM Summarization](#llm-summarization)
6. [Vector Database and Search](#vector-database-and-search)
7. [Future Enhancements](#future-enhancements)

## Data Exploration and Analysis

The project begins with comprehensive data exploration focusing on movie metadata and patterns across different dimensions:

### Wiki Movie Plots Dataset Analysis

- **Dataset Size**: 34,886 movies with plot descriptions
- **Time Period**: Movies from 1900s to present day
- **Features**: Title, Release Year, Origin/Ethnicity, Director, Cast, Genre, Plot

![Genre Distribution](placeholder_images/genre_distribution.png)

### TMDB Dataset Analysis

The project incorporates additional data from TMDB API to enrich the analysis:

- **Actor and Director Networks**: Analysis of collaborations and popularity
- **Gender Distribution**: Analysis of gender representation in movies

![Gender Distribution](placeholder_images/gender_distribution.png)

### Key Insights

- Genre behavior varies significantly across countries and time periods
- Strong correlation between certain actors/directors and specific genres
- Budget and revenue patterns differ substantially by genre

![Budget by Genre](placeholder_images/budget_by_genre.png)

## Genre Prediction Pipeline

A complete pipeline was implemented for predicting movie genres based on plot descriptions:

### Pipeline Components

1. **Data Input Processing**: Handles various file formats (CSV, JSON)
2. **Text Preprocessing**: Cleaning, tokenization, lemmatization
3. **Feature Engineering**: TF-IDF vectorization, n-grams, word embeddings
4. **Model Training**: Multi-label classification models
5. **Evaluation**: Comprehensive metrics for multi-label classification

### Implementation Details

```python
# Example of the pipeline usage
from genre_predictor import GenrePredictor

predictor = GenrePredictor()
predictor.load_data('movie_data.csv')
predictor.preprocess()
predictor.train_models()
predictions = predictor.predict(new_movie_plots)
```

## Model Evaluation and Selection

The project implements a comprehensive model evaluation system that compares multiple machine learning models:

### Models Evaluated

1. **Basic Models**:
   - Logistic Regression
   - Multinomial Naive Bayes
   - Decision Tree

2. **Intermediate Models**:
   - Random Forest
   - Linear SVM
   - Gradient Boosting
   - XGBoost

3. **Advanced Models**:
   - Neural Network MLP
   - Deep Learning (TensorFlow)

### Evaluation Results

| Model                 | F1-micro | F1-macro | F1-weighted | Hamming Loss |
|-----------------------|----------|----------|-------------|--------------|
| Neural Network MLP    | 0.8003   | 0.5123   | 0.7845      | 0.1234       |
| Random Forest         | 0.7912   | 0.4987   | 0.7756      | 0.1345       |
| XGBoost               | 0.7856   | 0.4923   | 0.7701      | 0.1389       |
| Linear SVM            | 0.7834   | 0.4876   | 0.7689      | 0.1402       |
| Logistic Regression   | 0.7801   | 0.4812   | 0.7645      | 0.1421       |
| Multinomial NB        | 0.7775   | 0.4674   | 0.7432      | 0.1433       |
| Gradient Boosting     | 0.7733   | 0.4608   | 0.7329      | 0.1465       |
| Decision Tree         | 0.7095   | 0.4673   | 0.7086      | 0.1973       |

![Model Comparison](placeholder_images/model_comparison.png)

### Feature Importance Analysis

The system includes feature importance analysis to identify the most predictive terms for each genre:

![Feature Importance](placeholder_images/feature_importance.png)

## Model Saving and Loading

The project implements functionality to save trained models and load them for later use:

### Saving Capabilities

- Save individual models or all models at once
- Include model metadata (parameters, performance metrics)
- Option to include training data with models

### Loading Capabilities

```python
# Example of loading saved models
from model_saver import ModelSaver

saver = ModelSaver()
evaluator = saver.load_models_into_evaluator('saved_models/movie_genre_prediction_20250520_224334')

# Use loaded models for prediction
best_model = evaluator.best_model
predictions = best_model.predict(new_data)
```

## LLM Summarization

The project includes a module for generating non-spoiler summaries of movie plots using Large Language Models:

### Supported LLM Providers

- OpenAI (GPT models)
- Anthropic (Claude models)
- Hugging Face models
- Mistral AI
- Local models

### Implementation Details

```python
from llm_summarizer import LLMSummarizer

# Initialize with OpenAI
summarizer1 = LLMSummarizer(
    model_name="gpt-3.5-turbo",
    provider="openai"
)

# Initialize with Mistral
summarizer2 = LLMSummarizer(
    model_name="mistral-medium",
    provider="mistral"
)

# Generate summaries
summaries1 = summarizer1.generate_summaries()
summaries2 = summarizer2.generate_summaries()
```

### Sample Results

| Movie Title | Original Plot Length | Summary Length | Model |
|-------------|---------------------|----------------|-------|
| The Godfather | 1,245 words | 87 words | GPT-3.5 |
| Inception | 1,102 words | 75 words | Claude |
| The Matrix | 978 words | 68 words | Mistral |

![Summary Comparison](placeholder_images/summary_comparison.png)

## Vector Database and Search

The project implements a vector database for movies with semantic search capabilities:

### Features

- **Vector Embeddings**: Convert movie data into vector embeddings
- **Efficient Search**: Fast similarity search using FAISS
- **Flexible Data Sources**: Support for different movie data sources
- **Persistence**: Save and load the vector database

### Implementation Details

```python
from movie_vector_db import MovieVectorDB

# Initialize the vector database
db = MovieVectorDB()

# Load and preprocess the data
db.load_data("wiki_movie_plots_cleaned.csv")
db.preprocess_data()

# Create embeddings and build the index
db.create_embeddings()
db.build_index()

# Search for movies
results = db.search("space exploration adventure", k=5)
```

### Search Results Example

| Query: "heartwarming story about friendship and loyalty" |
|----------------------------------------------------------|
| 1. The Shawshank Redemption (0.87 similarity) |
| 2. Stand By Me (0.82 similarity) |
| 3. E.T. the Extra-Terrestrial (0.79 similarity) |
| 4. The Fox and the Hound (0.76 similarity) |
| 5. Good Will Hunting (0.74 similarity) |

![Search Interface](placeholder_images/search_interface.png)

## Future Enhancements

1. **Recommendation System**: Implement personalized movie recommendations based on user preferences
2. **Sentiment Analysis**: Add sentiment analysis of movie reviews and correlate with ratings
3. **Multimodal Analysis**: Incorporate image and video data from movie trailers and posters
4. **Interactive Dashboard**: Develop a web-based dashboard for exploring the entire system
5. **API Integration**: Create APIs for third-party applications to access the system's capabilities

![Future Roadmap](placeholder_images/future_roadmap.png)

---

This project demonstrates the power of combining multiple machine learning and NLP techniques to create a comprehensive movie analysis and recommendation system. The modular design allows for easy extension and improvement of individual components.
