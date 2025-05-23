"""
NER Model for Movie Entity Extraction

This module provides functionality to:
1. Generate training data for movie entity extraction
2. Train NER models to extract Directors, Cast, and Genres from user queries
3. Save and load trained models
4. Extract entities from new text

Entity Types:
- DIRECTOR: Movie directors (e.g., "Christopher Nolan")
- CAST: Actors and actresses (e.g., "Leonardo DiCaprio") 
- GENRE: Movie genres (e.g., "action", "comedy")
"""

import os
import json
import random
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
from spacy.scorer import Scorer
from datetime import datetime
import logging

# Import existing utilities
from Utils import known_genres

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MovieNERDataGenerator:
    """Generates training data for movie NER model."""
    
    def __init__(self, movie_data_path: Optional[str] = None):
        """Initialize the data generator."""
        self.movie_data_path = movie_data_path
        self.directors = []
        self.cast_members = []
        self.genres = list(known_genres.keys())
        
        if movie_data_path and os.path.exists(movie_data_path):
            self.load_movie_data()
        else:
            self._load_fallback_data()
    
    def load_movie_data(self):
        """Load movie data and extract entities."""
        try:
            df = pd.read_csv(self.movie_data_path)
            
            # Extract directors
            if 'Director' in df.columns:
                directors = df['Director'].dropna().unique()
                self.directors = [d.strip() for d in directors if isinstance(d, str) and d.strip()]
            
            # Extract cast members
            if 'Cast' in df.columns:
                cast_data = df['Cast'].dropna()
                all_cast = []
                for cast_str in cast_data:
                    if isinstance(cast_str, str):
                        cast_list = [c.strip() for c in cast_str.split(',')]
                        all_cast.extend(cast_list)
                self.cast_members = list(set([c for c in all_cast if c and len(c) > 2]))
            
            logger.info(f"Loaded {len(self.directors)} directors and {len(self.cast_members)} cast members")
            
        except Exception as e:
            logger.error(f"Error loading movie data: {e}")
            self._load_fallback_data()
    
    def _load_fallback_data(self):
        """Load fallback data when movie dataset is not available."""
        self.directors = [
            "Christopher Nolan", "Quentin Tarantino", "Martin Scorsese", "Steven Spielberg",
            "Ridley Scott", "David Fincher", "Denis Villeneuve", "Jordan Peele",
            "Greta Gerwig", "Rian Johnson", "Chloe Zhao", "Barry Jenkins"
        ]
        
        self.cast_members = [
            "Leonardo DiCaprio", "Meryl Streep", "Robert De Niro", "Scarlett Johansson",
            "Tom Hanks", "Jennifer Lawrence", "Brad Pitt", "Angelina Jolie",
            "Will Smith", "Emma Stone", "Ryan Gosling", "Margot Robbie",
            "Denzel Washington", "Viola Davis", "Michael B. Jordan", "Lupita Nyong'o"
        ]
    
    def generate_training_data(self, num_samples: int = 1000) -> List[Tuple[str, Dict]]:
        """Generate training samples with entity annotations."""
        templates = [
            # Director patterns
            ("I want movies directed by {director}", [("director", "DIRECTOR")]),
            ("Show me films directed by {director}", [("director", "DIRECTOR")]),
            ("Find movies by {director}", [("director", "DIRECTOR")]),
            
            # Cast patterns
            ("I want movies with {actor}", [("actor", "CAST")]),
            ("Show me films starring {actor}", [("actor", "CAST")]),
            ("Find movies with {actor1} and {actor2}", [("actor1", "CAST"), ("actor2", "CAST")]),
            
            # Genre patterns
            ("I want {genre} movies", [("genre", "GENRE")]),
            ("Show me {genre} films", [("genre", "GENRE")]),
            ("Find {genre1} and {genre2} movies", [("genre1", "GENRE"), ("genre2", "GENRE")]),
            
            # Combined patterns
            ("I want {genre} movies directed by {director}", [("genre", "GENRE"), ("director", "DIRECTOR")]),
            ("Show me {genre} films with {actor}", [("genre", "GENRE"), ("actor", "CAST")]),
            ("Find {genre} movies starring {actor}", [("genre", "GENRE"), ("actor", "CAST")]),
        ]
        
        samples = []
        for _ in range(num_samples):
            template, entity_specs = random.choice(templates)
            text, entities = self._fill_template(template, entity_specs)
            if entities:  # Only add if we have valid entities
                annotations = {"entities": entities}
                samples.append((text, annotations))
        
        return samples
    
    def _fill_template(self, template: str, entity_specs: List[Tuple]) -> Tuple[str, List[Tuple]]:
        """Fill template with entities and create annotations."""
        filled_values = {}
        used_values = set()
        
        # Fill template with unique entities
        for entity_name, entity_type in entity_specs:
            if entity_type == "DIRECTOR":
                entity_list = self.directors
            elif entity_type == "CAST":
                entity_list = self.cast_members
            elif entity_type == "GENRE":
                entity_list = self.genres
            else:
                continue
            
            # Find unique value
            attempts = 0
            while attempts < 10:
                value = random.choice(entity_list)
                if value not in used_values:
                    filled_values[entity_name] = value
                    used_values.add(value)
                    break
                attempts += 1
            else:
                filled_values[entity_name] = random.choice(entity_list)
        
        # Fill the template
        filled_text = template.format(**filled_values)
        
        # Create entity annotations
        entities = []
        used_positions = set()
        
        for entity_name, entity_type in entity_specs:
            if entity_name not in filled_values:
                continue
                
            entity_value = filled_values[entity_name]
            start_pos = 0
            
            while True:
                pos = filled_text.find(entity_value, start_pos)
                if pos == -1:
                    break
                
                end_pos = pos + len(entity_value)
                
                # Check for overlaps
                overlaps = False
                for used_start, used_end in used_positions:
                    if not (end_pos <= used_start or pos >= used_end):
                        overlaps = True
                        break
                
                if not overlaps:
                    entities.append((pos, end_pos, entity_type))
                    used_positions.add((pos, end_pos))
                    break
                
                start_pos = pos + 1
        
        return filled_text, sorted(entities, key=lambda x: x[0])


class MovieNERModel:
    """Main class for training and using movie NER models."""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize the NER model."""
        self.model_name = model_name
        self.nlp = None
        self.ner = None
        self.entity_labels = ["DIRECTOR", "CAST", "GENRE"]
        self.training_data = []
        self.validation_data = []
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the spaCy model."""
        try:
            self.nlp = spacy.load(self.model_name)
            logger.info(f"Loaded base model: {self.model_name}")
        except (OSError, ValueError) as e:
            logger.warning(f"Could not load {self.model_name}: {e}")
            logger.info("Creating blank English model...")
            self.nlp = spacy.blank("en")
            if "tok2vec" not in self.nlp.pipe_names:
                self.nlp.add_pipe("tok2vec")
        
        # Add NER component
        if "ner" not in self.nlp.pipe_names:
            self.ner = self.nlp.add_pipe("ner", last=True)
        else:
            self.ner = self.nlp.get_pipe("ner")
        
        # Add entity labels
        for label in self.entity_labels:
            self.ner.add_label(label)
    
    def prepare_training_data(self, num_samples: int = 1000, movie_data_path: Optional[str] = None):
        """Generate and prepare training data."""
        logger.info(f"Generating {num_samples} training samples...")
        
        generator = MovieNERDataGenerator(movie_data_path)
        samples = generator.generate_training_data(num_samples)
        
        # Validate and split data
        valid_samples = self._validate_samples(samples)
        split_idx = int(0.8 * len(valid_samples))
        
        self.training_data = valid_samples[:split_idx]
        self.validation_data = valid_samples[split_idx:]
        
        logger.info(f"Prepared {len(self.training_data)} training and {len(self.validation_data)} validation samples")
    
    def _validate_samples(self, samples: List[Tuple]) -> List[Tuple]:
        """Validate training samples for overlapping entities."""
        valid_samples = []
        
        for text, annotations in samples:
            entities = annotations.get('entities', [])
            entities_sorted = sorted(entities, key=lambda x: x[0])
            
            is_valid = True
            # Check for overlaps
            for i in range(len(entities_sorted) - 1):
                current_start, current_end, _ = entities_sorted[i]
                next_start, _, _ = entities_sorted[i + 1]
                if current_end > next_start:
                    is_valid = False
                    break
            
            # Check bounds
            if is_valid:
                for start, end, _ in entities:
                    if start < 0 or end > len(text) or start >= end:
                        is_valid = False
                        break
            
            if is_valid:
                valid_samples.append((text, annotations))
        
        return valid_samples
    
    def train(self, n_iter: int = 30) -> Dict[str, Any]:
        """Train the NER model."""
        if not self.training_data:
            raise ValueError("No training data available. Call prepare_training_data() first.")
        
        logger.info(f"Training model for {n_iter} iterations...")
        
        # Disable other pipes during training
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "ner"]
        
        training_losses = []
        validation_scores = []
        
        with self.nlp.disable_pipes(*other_pipes):
            # Initialize
            try:
                optimizer = self.nlp.initialize()
            except AttributeError:
                optimizer = self.nlp.begin_training()
            
            for iteration in range(n_iter):
                random.shuffle(self.training_data)
                losses = {}
                
                # Create examples
                examples = []
                for text, annotations in self.training_data:
                    doc = self.nlp.make_doc(text)
                    try:
                        example = Example.from_dict(doc, annotations)
                        examples.append(example)
                    except ValueError:
                        continue
                
                # Train in batches
                batches = minibatch(examples, size=compounding(4.0, 32.0, 1.001))
                for batch in batches:
                    self.nlp.update(batch, drop=0.5, losses=losses, sgd=optimizer)
                
                training_losses.append(losses.get("ner", 0))
                
                # Evaluate every 5 iterations
                if iteration % 5 == 0:
                    val_score = self._evaluate()
                    validation_scores.append(val_score)
                    logger.info(f"Iteration {iteration}: Loss={losses.get('ner', 0):.4f}, "
                              f"Val F1={val_score}")
        
        final_score = self._evaluate()
        logger.info(f"Training completed. Final F1 score: {final_score.get('ents_f', 0):.4f}")
        
        return {
            'training_losses': training_losses,
            'validation_scores': validation_scores,
            'final_score': final_score,
            'n_iter': n_iter
        }
    
    def _evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation data."""
        if not self.validation_data:
            return {}
        
        examples = []
        for text, annotations in self.validation_data:
            doc = self.nlp.make_doc(text)
            try:
                example = Example.from_dict(doc, annotations)
                examples.append(example)
            except ValueError:
                continue
        
        if not examples:
            return {}
        
        scorer = Scorer()
        scores = scorer.score(examples)
        
        return {
            'ents_f': scores.get('ents_f', 0),
            'ents_p': scores.get('ents_p', 0),
            'ents_r': scores.get('ents_r', 0)
        }
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text."""
        if self.nlp is None:
            raise ValueError("Model not initialized")
        
        doc = self.nlp(text)
        
        entities = {'DIRECTOR': [], 'CAST': [], 'GENRE': []}
        
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)
        
        return entities
    
    def save_model(self, model_path: str) -> str:
        """Save the trained model."""
        os.makedirs(model_path, exist_ok=True)
        self.nlp.to_disk(model_path)
        
        # Save metadata
        metadata = {
            'entity_labels': self.entity_labels,
            'base_model': self.model_name,
            'training_samples': len(self.training_data),
            'validation_samples': len(self.validation_data),
            'saved_at': datetime.now().isoformat()
        }
        
        with open(os.path.join(model_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
        return model_path
    
    def load_model(self, model_path: str):
        """Load a trained model."""
        self.nlp = spacy.load(model_path)
        self.ner = self.nlp.get_pipe("ner")
        
        # Load metadata
        metadata_path = os.path.join(model_path, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.entity_labels = metadata.get('entity_labels', self.entity_labels)
        
        logger.info(f"Model loaded from {model_path}")


def train_movie_ner_model(movie_data_path: str = None, 
                         num_samples: int = 1000, 
                         n_iter: int = 30,
                         model_save_path: str = "saved_models/movie_ner_model") -> str:
    """
    Complete function to train a movie NER model.
    
    Args:
        movie_data_path: Path to movie dataset (optional)
        num_samples: Number of training samples to generate
        n_iter: Number of training iterations
        model_save_path: Path to save the trained model
        
    Returns:
        Path where the model was saved
    """
    # Initialize model
    model = MovieNERModel()
    
    # Prepare training data
    model.prepare_training_data(num_samples, movie_data_path)
    
    # Train the model
    metrics = model.train(n_iter)
    
    # Save the model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"{model_save_path}_{timestamp}"
    model.save_model(save_path)
    
    logger.info(f"Training completed. Model saved to: {save_path}")
    logger.info(f"Final F1 score: {metrics['final_score'].get('ents_f', 0):.4f}")
    
    return save_path
