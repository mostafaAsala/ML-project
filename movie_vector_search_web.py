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

import traceback
import streamlit as st
import os
import re
from typing import List, Dict
from movie_vector_db import MovieVectorDB
from ner_model import MovieNERModel

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


def display_search_results(results: List[Dict], entities: Dict[str, List[str]] = None):
    """Display search results with entity highlighting."""
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

                if 'genre' in movie and movie['genre']:
                    movie_genres = movie['genre']

                    # Handle different genre formats
                    if isinstance(movie_genres, list):
                        genre_text = ', '.join(str(g) for g in movie_genres)
                    elif isinstance(movie_genres, str):
                        genre_text = movie_genres
                    else:
                        genre_text = str(movie_genres)

                    # Highlight matched genres
                    if entities and entities['GENRE']:
                        for genre in entities['GENRE']:
                            if genre.lower() in genre_text.lower():
                                # Use case-insensitive replacement
                                import re
                                pattern = re.compile(re.escape(genre), re.IGNORECASE)
                                genre_text = pattern.sub(f"**{genre}**", genre_text)

                    st.write(f"**Genre:** {genre_text}")

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
                # Display plot
                if 'plot' in movie and movie['plot']:
                    st.write("**Plot:**")
                    st.write(movie['plot'])

                # Display wiki link if available
                if 'wiki_url' in movie and movie['wiki_url']:
                    st.write(f"[Wiki Page]({movie['wiki_url']})")


def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(
        page_title="Movie Vector Search with NER",
        page_icon="🎬",
        layout="wide"
    )

    st.title("🎬 Movie Vector Search with NER")
    st.write("""
    Search for movies using natural language queries. This app uses:
    - **Vector database** for semantic similarity search
    - **NER (Named Entity Recognition)** to extract and filter by directors, cast, and genres
    """)

    # Load NER model
    ner_model = load_ner_model()

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

                            # Step 4: Display results
                            if results:
                                display_search_results(results, entities if use_ner_filtering else None)
                            else:
                                st.warning("No results found after filtering. Try a broader query or disable NER filtering.")
                        else:
                            st.warning("No results found.")
                else:
                    st.info("Please enter a search query.")

            # NER Model Information
            st.sidebar.subheader("🎯 NER Model")
            if ner_model:
                st.sidebar.success("✓ NER model loaded")
                st.sidebar.write("**Extracts:**")
                st.sidebar.write("• Directors")
                st.sidebar.write("• Cast members")
                st.sidebar.write("• Genres")
            else:
                st.sidebar.error("✗ NER model not available")

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
            st.error(f"Error loading database: {traceback.format_exc()}")
    else:
        st.info("Please create a vector database using the sidebar options.")

        # Show NER demo even without database
        if ner_model:
            st.subheader("🎯 NER Demo (No Database Required)")
            st.write("Try the NER entity extraction while the database is being created:")

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

if __name__ == "__main__":
    main()
