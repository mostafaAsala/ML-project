"""
Movie Vector Search Web Interface with NER Integration

This script provides an enhanced web interface for the movie vector database search functionality.
It uses Streamlit to create an interactive web application with NER-based filtering.

Features:
- Semantic search using vector database
- NER-based entity extraction and filtering
- Advanced filtering by genre, cast, and directors
- Interactive results display

Usage:
    streamlit run movie_vector_search_web.py
"""

import streamlit as st
import os
import re
from typing import List, Dict
from movie_vector_db import MovieVectorDB
from ner_model import MovieNERModel
from genre_predictor import GenrePredictor
from seq2seq_summarizer_trainer import Seq2SeqSummarizerTrainer

# Add streamlit to requirements.txt if not already there
try:
    import streamlit
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "streamlit"])
    import streamlit


@st.cache_resource
def load_ner_model():
    """Load the NER model with caching for better performance."""
    # Try to find an existing trained NER model
    ner_model_paths = [
        "saved_models/complete_ner_model",
        "saved_models/demo_ner_model",
        "saved_models/notebook_ner_model"
    ]

    # Look for any NER model with timestamp
    import glob
    for base_path in ner_model_paths:
        pattern = f"{base_path}_*"
        matches = glob.glob(pattern)
        if matches:
            # Use the most recent one
            latest_model = max(matches)
            try:
                model = MovieNERModel()
                model.load_model(latest_model)
                st.success(f"✓ Loaded NER model from: {latest_model}")
                return model
            except Exception as e:
                st.warning(f"Failed to load NER model from {latest_model}: {e}")
                continue

    # If no trained model found, create and train a quick one
    st.warning("No trained NER model found. Training a quick model...")
    try:
        from ner_model import train_movie_ner_model

        with st.spinner("Training NER model... This may take a few minutes."):
            model_path = train_movie_ner_model(
                num_samples=300,  # Quick training
                n_iter=15,
                model_save_path="saved_models/streamlit_ner_model"
            )

            model = MovieNERModel()
            model.load_model(model_path)
            st.success(f"✓ Trained and loaded new NER model: {model_path}")
            return model

    except Exception as e:
        st.error(f"Failed to train NER model: {e}")
        return None


@st.cache_resource
def load_genre_predictor():
    """Load the genre predictor model with caching."""
    try:
        predictor = GenrePredictor.load(models_dir='saved_models')
        st.success("✓ Loaded genre predictor model")
        return predictor
    except Exception as e:
        st.warning(f"Could not load genre predictor: {e}")
        return None


@st.cache_resource
def load_summarizer():
    """Load the seq2seq summarizer with caching."""
    # Try to find an existing trained seq2seq model
    seq2seq_model_paths = [
        "seq2seq_summarizer_model",
        "saved_models/seq2seq_summarizer_model",
        "saved_models/seq2seq_model"
    ]

    # Look for any seq2seq model with timestamp
    import glob
    for base_path in seq2seq_model_paths:
        if os.path.exists(base_path):
            try:
                summarizer = Seq2SeqSummarizerTrainer.load_model(base_path)
                st.success(f"✓ Loaded seq2seq summarizer from: {base_path}")
                return summarizer
            except Exception as e:
                st.warning(f"Failed to load seq2seq model from {base_path}: {e}")
                continue

        # Also check for timestamped versions
        pattern = f"{base_path}_*"
        matches = glob.glob(pattern)
        if matches:
            latest_model = max(matches)
            try:
                summarizer = Seq2SeqSummarizerTrainer.load_model(latest_model)
                st.success(f"✓ Loaded seq2seq summarizer from: {latest_model}")
                return summarizer
            except Exception as e:
                st.warning(f"Failed to load seq2seq model from {latest_model}: {e}")
                continue

    # If no trained model found, try to create a basic one
    st.warning("No trained seq2seq model found. Using base T5 model (limited performance).")
    try:
        summarizer = Seq2SeqSummarizerTrainer(
            model_name="t5-small",
            max_input_length=512,
            max_output_length=128
        )
        st.warning("✓ Loaded base T5 model (not fine-tuned for movie summaries)")
        return summarizer
    except Exception as e:
        st.error(f"Could not load any summarizer: {e}")
        return None


def predict_movie_genres(movie: Dict, genre_predictor: GenrePredictor) -> Dict[str, List[str]]:
    """Predict genres for any movie and return both original and predicted genres."""
    result = {
        'original_genres': [],
        'predicted_genres': []
    }

    if not genre_predictor:
        return result

    # Get existing genres
    existing_genres = movie.get('genre', [])
    if existing_genres:
        if isinstance(existing_genres, list):
            result['original_genres'] = existing_genres
        else:
            result['original_genres'] = [existing_genres]

    # Always predict genres regardless of whether movie has them
    plot = movie.get('plot', '')
    origin = movie.get('origin', '') or movie.get('Origin/Ethnicity', '') or 'Unknown'

    if not plot:
        return result

    try:
        # Create data in the format expected by the predictor
        prediction_data = {
            'plot_lemmatized': plot,  # Use plot as is, assuming it's already processed
            'Origin/Ethnicity': origin
        }

        # Make prediction
        predicted_genres = genre_predictor.predict(prediction_data)

        # Handle different return formats
        if isinstance(predicted_genres, list) and len(predicted_genres) > 0:
            if isinstance(predicted_genres[0], list):
                result['predicted_genres'] = predicted_genres[0]
            else:
                result['predicted_genres'] = predicted_genres

        return result

    except Exception as e:
        st.warning(f"Error predicting genres for {movie.get('title', 'unknown movie')}: {e}")
        return result


def enhance_results_with_predictions(results: List[Dict], genre_predictor) -> List[Dict]:
    """Enhance search results with genre predictions for all movies."""
    if not genre_predictor or not results:
        return results

    enhanced_results = []

    # Show progress for genre prediction
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, movie in enumerate(results):
        # Update progress
        progress = (i + 1) / len(results)
        progress_bar.progress(progress)
        status_text.text(f"🤖 Predicting genres... {i+1}/{len(results)}")

        # Create enhanced movie copy
        enhanced_movie = movie.copy()

        # Predict genres
        genre_predictions = predict_movie_genres(movie, genre_predictor)
        enhanced_movie['genre_predictions'] = genre_predictions

        enhanced_results.append(enhanced_movie)

    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()

    return enhanced_results


def summarize_plot(plot: str, summarizer: Seq2SeqSummarizerTrainer, title: str = "") -> str:
    """Summarize a movie plot using the seq2seq summarizer."""
    if not summarizer or not plot:
        return plot[:200] + "..." if len(plot) > 200 else plot

    try:
        # For seq2seq models, we can directly pass the plot text
        # The model should be trained to generate summaries from plot text

        # Prepare the input text - some models work better with a prefix
        if title:
            input_text = f"summarize: {title}: {plot}"
        else:
            input_text = f"summarize: {plot}"

        # Generate summary using the seq2seq model
        summary = summarizer.generate_summary(input_text)

        # Clean up the summary
        summary = summary.strip()

        # If summary is too short or seems like it failed, fallback to truncation
        if len(summary) < 10 or summary.lower().startswith("summarize"):
            return plot[:200] + "..." if len(plot) > 200 else plot

        return summary

    except Exception as e:
        # Fallback to simple truncation
        return plot[:200] + "..." if len(plot) > 200 else plot


def extract_entities_from_query(query: str, ner_model: MovieNERModel) -> Dict[str, List[str]]:
    """Extract entities from user query using NER model."""
    if ner_model is None:
        return {'DIRECTOR': [], 'CAST': [], 'GENRE': []}

    try:
        entities = ner_model.extract_entities(query)
        return entities
    except Exception as e:
        st.error(f"Error extracting entities: {e}")
        return {'DIRECTOR': [], 'CAST': [], 'GENRE': []}


def filter_results_by_entities(results: List[Dict], entities: Dict[str, List[str]]) -> List[Dict]:
    """Filter search results based on extracted entities."""
    if not any(entities.values()):
        return results  # No entities to filter by

    filtered_results = []

    for movie in results:
        should_include = True

        # Check director filter
        if entities['DIRECTOR']:
            movie_director = movie.get('director', '').lower()
            if not any(director.lower() in movie_director for director in entities['DIRECTOR']):
                should_include = False

        # Check cast filter
        if entities['CAST'] and should_include:
            movie_cast = movie.get('cast', '') or movie.get('cast_names', '')

            if isinstance(movie_cast, list):
                # Handle case where cast is a list
                cast_text = ' '.join(str(c).lower() for c in movie_cast)
            elif isinstance(movie_cast, str):
                cast_text = movie_cast.lower()
            else:
                cast_text = str(movie_cast).lower()

            if not any(actor.lower() in cast_text for actor in entities['CAST']):
                should_include = False

        # Check genre filter
        if entities['GENRE'] and should_include:
            movie_genres = movie.get('genre', [])
            if isinstance(movie_genres, str):
                # Handle case where genre is a string (comma-separated)
                movie_genres = [g.strip().lower() for g in movie_genres.split(',')]
            elif isinstance(movie_genres, list):
                # Handle case where genre is already a list
                movie_genres = [g.lower() if isinstance(g, str) else str(g).lower() for g in movie_genres]
            else:
                movie_genres = []

            if not any(genre.lower() in ' '.join(movie_genres) for genre in entities['GENRE']):
                should_include = False

        if should_include:
            filtered_results.append(movie)

    return filtered_results


def display_entity_info(entities: Dict[str, List[str]]):
    """Display extracted entities in a nice format."""
    if any(entities.values()):
        st.subheader("🎯 Extracted Entities")

        col1, col2, col3 = st.columns(3)

        with col1:
            if entities['DIRECTOR']:
                st.write("**Directors:**")
                for director in entities['DIRECTOR']:
                    st.write(f"• {director}")

        with col2:
            if entities['CAST']:
                st.write("**Cast:**")
                for actor in entities['CAST']:
                    st.write(f"• {actor}")

        with col3:
            if entities['GENRE']:
                st.write("**Genres:**")
                for genre in entities['GENRE']:
                    st.write(f"• {genre}")

        st.write("---")


def display_search_results(results: List[Dict], entities: Dict[str, List[str]] = None,
                          summarizer=None):
    """Display search results with entity highlighting, genre prediction, and plot summarization."""
    if not results:
        st.warning("No results found.")
        return

    # Show filtering info
    if entities and any(entities.values()):
        total_entities = sum(len(v) for v in entities.values())
        st.info(f"🔍 Results filtered by {total_entities} extracted entities")

    st.success(f"Found {len(results)} results")

    # Display results
    for i, movie in enumerate(results):
        with st.expander(f"{i+1}. {movie['title']} (Score: {movie['similarity_score']:.2f})"):
            col1, col2 = st.columns([1, 3])

            with col1:
                # Display movie metadata with entity highlighting
                if 'year' in movie and movie['year']:
                    st.write(f"**Year:** {movie['year']}")

                # Handle genres - show both original and predicted
                original_genres = movie.get('genre', [])
                genre_predictions = movie.get('genre_predictions', {})
                predicted_genres = genre_predictions.get('predicted_genres', [])

                # Display original genres
                if original_genres:
                    if isinstance(original_genres, list):
                        original_text = ', '.join(str(g) for g in original_genres)
                    else:
                        original_text = str(original_genres)

                    # Highlight matched genres
                    if entities and entities['GENRE']:
                        for genre in entities['GENRE']:
                            if genre.lower() in original_text.lower():
                                pattern = re.compile(re.escape(genre), re.IGNORECASE)
                                original_text = pattern.sub(f"**{genre}**", original_text)

                    st.write(f"**Genre:** {original_text}")

                # Display predicted genres if available
                if predicted_genres:
                    predicted_text = ', '.join(str(g) for g in predicted_genres)

                    # Highlight matched predicted genres
                    if entities and entities['GENRE']:
                        for genre in entities['GENRE']:
                            if genre.lower() in predicted_text.lower():
                                pattern = re.compile(re.escape(genre), re.IGNORECASE)
                                predicted_text = pattern.sub(f"**{genre}**", predicted_text)

                    if original_genres:
                        st.write(f"**AI Predicted:** {predicted_text} 🤖")
                    else:
                        st.write(f"**Genre:** {predicted_text} 🤖")
                        st.caption("🤖 = AI predicted (no original genres)")

                # If no genres at all
                if not original_genres and not predicted_genres:
                    st.write("**Genre:** Not available")

                if 'director' in movie and movie['director']:
                    director_text = movie['director']
                    # Highlight matched directors
                    if entities and entities['DIRECTOR']:
                        for director in entities['DIRECTOR']:
                            if director.lower() in director_text.lower():
                                # Use case-insensitive replacement
                                pattern = re.compile(re.escape(director), re.IGNORECASE)
                                director_text = pattern.sub(f"**{director}**", director_text)
                    st.write(f"**Director:** {director_text}")

                # Handle cast display
                cast_data = movie.get('cast') or movie.get('cast_names')
                if cast_data:
                    # Handle different cast formats
                    if isinstance(cast_data, list):
                        cast_text = ', '.join(str(c) for c in cast_data)
                    elif isinstance(cast_data, str):
                        cast_text = cast_data
                    else:
                        cast_text = str(cast_data)

                    # Highlight matched cast
                    if entities and entities['CAST']:
                        for actor in entities['CAST']:
                            if actor.lower() in cast_text.lower():
                                # Use case-insensitive replacement
                                pattern = re.compile(re.escape(actor), re.IGNORECASE)
                                cast_text = pattern.sub(f"**{actor}**", cast_text)

                    st.write(f"**Cast:** {cast_text}")

            with col2:
                # Display plot with summarization option
                if 'plot' in movie and movie['plot']:
                    plot_text = movie['plot']

                    # Show summarization options
                    col2a, col2b = st.columns([3, 1])

                    with col2a:
                        st.write("**Plot:**")

                    with col2b:
                        if summarizer and len(plot_text) > 300:
                            if st.button(f"📝 Summarize", key=f"summarize_{i}"):
                                with st.spinner("🤖 Summarizing plot..."):
                                    summary = summarize_plot(plot_text, summarizer, movie.get('title', ''))
                                    if summary != plot_text:
                                        st.session_state[f"summary_{i}"] = summary

                    # Display plot or summary
                    if f"summary_{i}" in st.session_state:
                        st.write(st.session_state[f"summary_{i}"])
                        st.caption("🤖 AI summarized")
                        if st.button(f"📖 Show full plot", key=f"full_plot_{i}"):
                            del st.session_state[f"summary_{i}"]
                            st.rerun()
                    else:
                        # Show truncated plot if too long
                        if len(plot_text) > 500:
                            st.write(plot_text[:500] + "...")
                            st.caption("Plot truncated. Use summarize button for AI summary.")
                        else:
                            st.write(plot_text)

                # Display wiki link if available
                if 'wiki_url' in movie and movie['wiki_url']:
                    st.write(f"[📖 Wiki Page]({movie['wiki_url']})")


def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(
        page_title="AI-Powered Movie Search",
        page_icon="🎬",
        layout="wide"
    )

    st.title("🎬 AI-Powered Movie Search")
    st.write("""
    Search for movies using natural language queries. This app uses multiple AI models:
    - **🔍 Vector database** for semantic similarity search
    - **🎯 NER (Named Entity Recognition)** to extract and filter by directors, cast, and genres
    - **🤖 Genre prediction** for movies missing genre information
    - **📝 Seq2Seq summarization** to create concise, non-spoiler summaries using fine-tuned T5/BART models
    """)

    # Load models
    ner_model = load_ner_model()
    genre_predictor = load_genre_predictor()
    summarizer = load_summarizer()

    # Sidebar for database operations
    st.sidebar.title("Database Operations")

    # Check if vector database exists
    db_exists = os.path.exists("saved_models/vector_db/faiss_index.bin")

    if db_exists:
        st.sidebar.success("Vector database found!")
    else:
        st.sidebar.warning("No vector database found. Please create one.")

    # Database creation options
    st.sidebar.subheader("Create New Database")
    data_source = st.sidebar.selectbox(
        "Data Source",
        ["wiki", "tmdb"],
        help="Select the source of movie data"
    )

    if data_source == "wiki":
        default_data_path = "wiki_movie_plots_cleaned.csv"
    else:
        default_data_path = "tmdb_5000_credits.csv"

    data_path = st.sidebar.text_input("Data Path", value=default_data_path)

    if st.sidebar.button("Create Database"):
        with st.spinner("Creating vector database... This may take a while."):
            try:
                db = MovieVectorDB()
                db.load_data(data_path, data_source=data_source)
                db.preprocess_data()
                db.create_embeddings()
                db.build_index()
                db.save()
                st.sidebar.success("Vector database created successfully!")
                db_exists = True
            except Exception as e:
                st.sidebar.error(f"Error creating database: {e}")

    # Main content area for search
    if db_exists:
        # Load the database
        try:
            db = MovieVectorDB()
            db.load()

            # Search interface
            st.subheader("🔍 Search for Movies")

            # Search options
            col1, col2 = st.columns([3, 1])

            with col1:
                # Initialize session state for query
                if 'query' not in st.session_state:
                    st.session_state.query = ""

                query = st.text_input(
                    "Enter your search query:",
                    value=st.session_state.query,
                    placeholder="E.g., action movies directed by Christopher Nolan with Leonardo DiCaprio",
                    help="Try queries like: 'sci-fi movies with Tom Hanks', 'comedy films by Quentin Tarantino', 'horror movies starring Lupita Nyong'o'"
                )

            with col2:
                num_results = st.slider("Results", min_value=1, max_value=20, value=10)
                use_ner_filtering = st.checkbox("🎯 NER Filtering", value=True,
                                              help="Use NER to extract and filter by entities")
                use_genre_prediction = st.checkbox("🤖 Genre Prediction", value=True,
                                                 help="Predict genres for all retrieved movies using AI")

            if st.button("🔍 Search", type="primary") or query:
                if query:
                    with st.spinner("Searching and analyzing query..."):
                        # Step 1: Extract entities from query using NER
                        entities = {'DIRECTOR': [], 'CAST': [], 'GENRE': []}
                        if use_ner_filtering and ner_model:
                            entities = extract_entities_from_query(query, ner_model)

                        # Step 2: Perform vector search
                        # Increase search results if we're going to filter
                        search_k = num_results * 3 if use_ner_filtering and any(entities.values()) else num_results
                        results = db.search(query, k=search_k)

                        if results:
                            # Step 3: Filter results by extracted entities
                            if use_ner_filtering and any(entities.values()):
                                original_count = len(results)
                                results = filter_results_by_entities(results, entities)

                                # Limit to requested number of results
                                results = results[:num_results]

                                # Show entity extraction results
                                display_entity_info(entities)

                                if len(results) < original_count:
                                    st.info(f"📊 Filtered from {original_count} to {len(results)} results based on extracted entities")
                            else:
                                results = results[:num_results]

                            # Step 4: Enhance results with genre predictions
                            if results and genre_predictor and use_genre_prediction:
                                st.info("🤖 Enhancing results with AI genre predictions...")
                                results = enhance_results_with_predictions(results, genre_predictor)

                            # Step 5: Display results
                            if results:
                                display_search_results(
                                    results,
                                    entities if use_ner_filtering else None,
                                    summarizer
                                )
                            else:
                                st.warning("No results found after filtering. Try a broader query or disable NER filtering.")
                        else:
                            st.warning("No results found.")
                else:
                    st.info("Please enter a search query.")

            # AI Models Information
            st.sidebar.subheader("🤖 AI Models")

            # NER Model
            if ner_model:
                st.sidebar.success("✓ NER model loaded")
                st.sidebar.write("**Extracts:** Directors, Cast, Genres")
            else:
                st.sidebar.error("✗ NER model not available")

            # Genre Predictor
            if genre_predictor:
                st.sidebar.success("✓ Genre predictor loaded")
                st.sidebar.write("**Predicts:** Genres for all movies")
                st.sidebar.write("**Shows:** Original + AI predicted")
            else:
                st.sidebar.warning("⚠ Genre predictor not available")

            # Seq2Seq Summarizer
            if summarizer:
                st.sidebar.success("✓ Seq2Seq summarizer loaded")
                st.sidebar.write("**Summarizes:** Long movie plots")
                st.sidebar.write("**Model:** Fine-tuned T5/BART")
            else:
                st.sidebar.warning("⚠ Seq2Seq summarizer not available")

            # Example queries
            st.sidebar.subheader("💡 Example Queries")
            example_queries = [
                "action movies directed by Christopher Nolan",
                "comedy films with Will Smith",
                "sci-fi movies starring Tom Hanks",
                "horror films by Jordan Peele",
                "animated movies for family",
                "thriller movies with Leonardo DiCaprio",
                "romantic comedies with Emma Stone",
                "drama films directed by Martin Scorsese"
            ]

            selected_example = st.sidebar.selectbox(
                "Try an example:",
                [""] + example_queries,
                help="Select an example query to test NER filtering"
            )

            if selected_example and st.sidebar.button("Use Example"):
                st.session_state.query = selected_example
                st.rerun()

            # Show database statistics
            st.sidebar.subheader("📊 Database Statistics")
            if hasattr(db, 'movies_df') and db.movies_df is not None:
                st.sidebar.write(f"Total movies: {len(db.movies_df)}")

                if 'genre' in db.movies_df.columns:
                    # Show genre distribution
                    genres = db.movies_df['genre'].str.split(',').explode().str.strip()
                    genre_counts = genres.value_counts().head(10)
                    st.sidebar.write("Top 10 Genres:")
                    st.sidebar.bar_chart(genre_counts)

                if 'year' in db.movies_df.columns:
                    # Show year distribution
                    year_counts = db.movies_df['year'].value_counts().sort_index()
                    year_counts = year_counts[year_counts.index > 1900]  # Filter out very old or invalid years
                    st.sidebar.write("Movies by Decade:")
                    st.sidebar.line_chart(year_counts.resample('10Y').sum())

        except Exception as e:
            st.error(f"Error loading database: {e}")
    else:
        st.info("Please create a vector database using the sidebar options.")

        # Show model demos even without database
        col1, col2, col3 = st.columns(3)

        with col1:
            # Show NER demo
            if ner_model:
                st.subheader("🎯 NER Demo")
                st.write("Try entity extraction:")

                demo_query = st.text_input(
                    "Test NER extraction:",
                    placeholder="E.g., action movies directed by Christopher Nolan",
                    help="Enter a query to see what entities the NER model can extract"
                )

                if demo_query:
                    entities = extract_entities_from_query(demo_query, ner_model)
                    if any(entities.values()):
                        display_entity_info(entities)
                    else:
                        st.info("No entities detected in this query.")

        with col2:
            # Show genre prediction demo
            if genre_predictor:
                st.subheader("🤖 Genre Prediction Demo")
                st.write("Try genre prediction:")

                demo_plot_genre = st.text_area(
                    "Test genre prediction:",
                    placeholder="Enter a movie plot to predict genres...",
                    help="Enter a movie plot to see what genres the AI predicts",
                    height=100,
                    key="genre_demo_plot"
                )

                if demo_plot_genre and len(demo_plot_genre) > 50:
                    if st.button("🤖 Predict Genres", key="predict_genres_demo"):
                        with st.spinner("🤖 Predicting genres..."):
                            demo_movie = {
                                'plot': demo_plot_genre,
                                'title': 'Demo Movie',
                                'origin': 'Unknown'
                            }
                            genre_predictions = predict_movie_genres(demo_movie, genre_predictor)
                            predicted_genres = genre_predictions.get('predicted_genres', [])

                            if predicted_genres:
                                st.write("**Predicted Genres:**")
                                st.write(', '.join(predicted_genres))
                                st.caption("🤖 Generated by AI genre predictor")
                            else:
                                st.info("No genres predicted for this plot.")
            else:
                st.subheader("🤖 Genre Predictor")
                st.write("No genre predictor loaded.")
                st.info("Train a genre predictor to see AI-powered genre predictions.")

        with col3:
            # Show seq2seq summarizer demo
            if summarizer:
                st.subheader("📝 Summarizer Demo")
                st.write("Try plot summarization:")

                demo_plot = st.text_area(
                    "Test plot summarization:",
                    placeholder="Enter a movie plot to summarize...",
                    help="Enter a movie plot to see the seq2seq summarizer in action",
                    height=100
                )

                if demo_plot and len(demo_plot) > 50:
                    if st.button("📝 Generate Summary"):
                        with st.spinner("🤖 Generating summary..."):
                            summary = summarize_plot(demo_plot, summarizer, "Demo Movie")
                            st.write("**Summary:**")
                            st.write(summary)
                            st.caption("🤖 Generated by seq2seq model")
            else:
                st.subheader("📝 Seq2Seq Model")
                st.write("No trained seq2seq model found.")
                st.info("""
                To use plot summarization:
                1. Train a seq2seq model using the example script
                2. The trained model will be automatically loaded
                """)

                if st.button("📖 View Training Instructions"):
                    st.code("""
# Generate training data and train model:
python seq2seq_summarizer_example.py --generate --train --data movie_data.csv --sample-size 500 --model t5-small --epochs 3
                    """, language="bash")

if __name__ == "__main__":
    main()
