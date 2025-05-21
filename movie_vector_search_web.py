"""
Movie Vector Search Web Interface

This script provides a simple web interface for the movie vector database search functionality.
It uses Streamlit to create an interactive web application.

Usage:
    streamlit run movie_vector_search_web.py
"""

import streamlit as st
import pandas as pd
import os
from movie_vector_db import MovieVectorDB

# Add streamlit to requirements.txt if not already there
try:
    import streamlit
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "streamlit"])
    import streamlit

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(
        page_title="Movie Vector Search",
        page_icon="🎬",
        layout="wide"
    )
    
    st.title("🎬 Movie Vector Search")
    st.write("""
    Search for movies using natural language queries. This app uses a vector database
    to find movies that are semantically similar to your query.
    """)
    
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
            st.subheader("Search for Movies")
            query = st.text_input("Enter your search query:", 
                                 placeholder="E.g., science fiction movie about time travel")
            
            num_results = st.slider("Number of results", min_value=1, max_value=20, value=5)
            
            if st.button("Search") or query:
                if query:
                    with st.spinner("Searching..."):
                        results = db.search(query, k=num_results)
                        
                        if results:
                            st.success(f"Found {len(results)} results")
                            
                            # Display results in a more visual way
                            for i, movie in enumerate(results):
                                with st.expander(f"{i+1}. {movie['title']} (Score: {movie['similarity_score']:.2f})"):
                                    col1, col2 = st.columns([1, 3])
                                    
                                    with col1:
                                        # Display movie metadata
                                        if 'year' in movie and movie['year']:
                                            st.write(f"**Year:** {movie['year']}")
                                        
                                        if 'genre' in movie and movie['genre']:
                                            st.write(f"**Genre:** {movie['genre']}")
                                        
                                        if 'director' in movie and movie['director']:
                                            st.write(f"**Director:** {movie['director']}")
                                        
                                        if 'cast' in movie and movie['cast']:
                                            st.write(f"**Cast:** {movie['cast']}")
                                        elif 'cast_names' in movie and movie['cast_names']:
                                            st.write(f"**Cast:** {movie['cast_names']}")
                                    
                                    with col2:
                                        # Display plot
                                        if 'plot' in movie and movie['plot']:
                                            st.write("**Plot:**")
                                            st.write(movie['plot'])
                                        
                                        # Display wiki link if available
                                        if 'wiki_url' in movie and movie['wiki_url']:
                                            st.write(f"[Wiki Page]({movie['wiki_url']})")
                        else:
                            st.warning("No results found.")
                else:
                    st.info("Please enter a search query.")
            
            # Show database statistics
            st.sidebar.subheader("Database Statistics")
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

if __name__ == "__main__":
    main()
