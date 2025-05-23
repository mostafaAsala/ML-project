# Movie NER Model

A Named Entity Recognition (NER) system for extracting movie-related entities from user text queries. This system can identify and extract directors, cast members, and genres from natural language input.

## 📋 Overview

The Movie NER Model provides functionality to:

- **Extract movie entities** from user queries: Directors, Cast, and Genres
- **Generate synthetic training data** automatically from movie datasets
- **Train custom NER models** using spaCy's NER capabilities
- **Save and load trained models** for reuse
- **Process single queries or batches** efficiently

## 🎯 Entity Types

The NER model recognizes three main entity types:

| Entity Type | Description | Examples |
|-------------|-------------|----------|
| **DIRECTOR** | Movie directors | "Christopher Nolan", "Quentin Tarantino" |
| **CAST** | Actors and actresses | "Leonardo DiCaprio", "Meryl Streep" |
| **GENRE** | Movie genres | "action", "comedy", "horror" |

## 📁 Files Structure

```
├── ner_model.py           # Core NER model and training functionality
├── ner_example.py         # Example usage and testing scripts
├── movie_ner_demo.ipynb   # Interactive Jupyter notebook
└── README_NER.md          # This documentation
```

## 🚀 Quick Start

### Option 1: Python Script
```bash
python ner_example.py
```

### Option 2: Jupyter Notebook
```bash
jupyter notebook movie_ner_demo.ipynb
```

### Option 3: Direct Usage
```python
from ner_model import train_movie_ner_model, MovieNERModel

# Train a model
model_path = train_movie_ner_model(num_samples=500, n_iter=25)

# Use the model
model = MovieNERModel()
model.load_model(model_path)
entities = model.extract_entities("I want action movies directed by Christopher Nolan")
print(entities)
# Output: {'DIRECTOR': ['Christopher Nolan'], 'CAST': [], 'GENRE': ['action']}
```

## 📦 Installation

### Prerequisites
```bash
pip install spacy pandas numpy
python -m spacy download en_core_web_sm
```

### Optional: For better performance
```bash
pip install spacy-lookups-data
```

## 🔧 Core Components

### 1. MovieNERDataGenerator
Generates synthetic training data for the NER model.

```python
from ner_model import MovieNERDataGenerator

# Create generator
generator = MovieNERDataGenerator('movie_data.csv')  # Optional movie dataset

# Generate training samples
samples = generator.generate_training_data(num_samples=1000)

# Each sample: (text, {"entities": [(start, end, label), ...]})
```

### 2. MovieNERModel
Main class for training and using NER models.

```python
from ner_model import MovieNERModel

# Initialize model
model = MovieNERModel()

# Prepare training data
model.prepare_training_data(num_samples=500)

# Train the model
metrics = model.train(n_iter=25)

# Extract entities
entities = model.extract_entities("Find horror movies with Tom Hanks")
```

### 3. Complete Training Function
One-line function to train a complete model.

```python
from ner_model import train_movie_ner_model

model_path = train_movie_ner_model(
    movie_data_path='wiki_movie_plots_deduped.csv',  # Optional
    num_samples=1000,
    n_iter=30,
    model_save_path="saved_models/my_ner_model"
)
```

## 💡 Usage Examples

### Basic Entity Extraction
```python
model = MovieNERModel()
model.load_model('saved_models/trained_model')

# Single query
entities = model.extract_entities("I want action movies directed by Christopher Nolan")
print(entities)
# {'DIRECTOR': ['Christopher Nolan'], 'CAST': [], 'GENRE': ['action']}

# Multiple queries
queries = [
    "Show me comedy films with Will Smith",
    "Find horror movies starring Lupita Nyong'o",
    "I love animated movies"
]

for query in queries:
    entities = model.extract_entities(query)
    print(f"{query} -> {entities}")
```

### Training with Custom Data
```python
# Custom training data
custom_data = [
    ("I want movies directed by James Cameron", 
     {"entities": [(27, 40, "DIRECTOR")]}),
    ("Show me films with Leonardo DiCaprio", 
     {"entities": [(20, 36, "CAST")]}),
    ("Find action movies", 
     {"entities": [(5, 11, "GENRE")]})
]

model = MovieNERModel()
model.training_data = custom_data[:2]
model.validation_data = custom_data[2:]
model.train(n_iter=20)
```

### Training with Movie Dataset
```python
# If you have a movie dataset CSV file
model_path = train_movie_ner_model(
    movie_data_path='wiki_movie_plots_deduped.csv',
    num_samples=2000,
    n_iter=40
)
```

## 📊 Training Data Templates

The system uses various templates to generate realistic training data:

### Director Templates
- "I want movies directed by {director}"
- "Show me films directed by {director}"
- "Find movies by {director}"

### Cast Templates
- "I want movies with {actor}"
- "Show me films starring {actor}"
- "Find movies with {actor1} and {actor2}"

### Genre Templates
- "I want {genre} movies"
- "Show me {genre} films"
- "Find {genre1} and {genre2} movies"

### Combined Templates
- "I want {genre} movies directed by {director}"
- "Show me {genre} films with {actor}"
- "Find {genre} movies starring {actor}"

## 🎯 Performance Tips

### Training Parameters
- **num_samples**: 500-2000 for good performance
- **n_iter**: 20-40 iterations usually sufficient
- **Use real movie data**: Improves entity recognition accuracy

### Model Performance
- **F1 Score**: Typically 0.7-0.9 with good training data
- **Training Time**: 2-10 minutes depending on data size
- **Memory Usage**: ~100-500MB for trained models

### Best Practices
1. **Validate training data** to avoid overlapping entities
2. **Use balanced entity types** in training data
3. **Save models** for reuse instead of retraining
4. **Test with diverse queries** to evaluate performance

## 🔍 Example Results

| User Query | Extracted Entities |
|------------|-------------------|
| "I want action movies directed by Christopher Nolan" | DIRECTOR: Christopher Nolan<br>GENRE: action |
| "Show me comedy films with Will Smith and Kevin Hart" | CAST: Will Smith, Kevin Hart<br>GENRE: comedy |
| "Find horror movies starring Lupita Nyong'o" | CAST: Lupita Nyong'o<br>GENRE: horror |
| "I love animated movies for family viewing" | GENRE: animated |

## 🛠️ Integration with Other Components

### With Genre Predictor
```python
# Extract entities from user query
entities = ner_model.extract_entities("I want sci-fi movies by Denis Villeneuve")

# Use extracted entities with genre predictor
if entities['GENRE']:
    predictions = genre_predictor.predict_by_genre(entities['GENRE'])
```

### With Vector Search
```python
# Extract entities
entities = ner_model.extract_entities("Find movies with Tom Hanks")

# Use with vector search
if entities['CAST']:
    similar_movies = vector_db.search(f"movies with {entities['CAST'][0]}")
```

## 📈 Model Evaluation

The training process provides comprehensive metrics:

```python
metrics = model.train(n_iter=30)

print(f"Final F1 Score: {metrics['final_score']['ents_f']:.4f}")
print(f"Precision: {metrics['final_score']['ents_p']:.4f}")
print(f"Recall: {metrics['final_score']['ents_r']:.4f}")
```

## 🗂️ Model Persistence

### Saving Models
```python
# Save during training
model_path = model.save_model("saved_models/my_ner_model")

# Or use the complete training function
model_path = train_movie_ner_model(model_save_path="saved_models/my_model")
```

### Loading Models
```python
model = MovieNERModel()
model.load_model("saved_models/my_ner_model_20240101_120000")
```

### Model Structure
```
saved_models/my_ner_model_20240101_120000/
├── config.cfg          # spaCy model configuration
├── meta.json           # spaCy model metadata  
├── tokenizer/          # Tokenizer component
├── ner/               # NER model files
└── metadata.json      # Custom metadata
```

## 🚨 Troubleshooting

### Common Issues

**1. spaCy Model Not Found**
```bash
python -m spacy download en_core_web_sm
```

**2. Overlapping Entity Errors**
- The system automatically validates and filters overlapping entities
- Check training data format if errors persist

**3. Low Performance**
- Increase `num_samples` (try 1000-2000)
- Increase `n_iter` (try 30-50)
- Use real movie dataset for training

**4. Memory Issues**
- Reduce batch size in training
- Use smaller training datasets
- Close other applications

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📝 File Descriptions

### `ner_model.py`
Core implementation containing:
- `MovieNERDataGenerator`: Training data generation
- `MovieNERModel`: Model training and entity extraction
- `train_movie_ner_model()`: Complete training pipeline

### `ner_example.py`
Example usage demonstrating:
- Complete training workflow
- Entity extraction examples
- Before/after training comparison
- Batch processing
- Custom training data

### `movie_ner_demo.ipynb`
Interactive Jupyter notebook with:
- Step-by-step training process
- Visual results and comparisons
- Multiple test scenarios
- Complete documentation

## 🔮 Future Enhancements

Potential improvements:
- **Additional entity types**: Production companies, release years
- **Multilingual support**: Non-English movie queries
- **Active learning**: Iterative improvement with user feedback
- **Confidence scores**: Entity extraction confidence levels
- **Fuzzy matching**: Handle misspelled names

## 📄 License

This NER model is part of the Movie Analysis ML Project and follows the same licensing terms.

## 🤝 Contributing

To contribute:
1. Test the model with your movie datasets
2. Report issues or suggest improvements
3. Share training results and performance metrics
4. Suggest new entity types or templates

---

**Happy entity extracting! 🎬🤖**
