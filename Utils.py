import re

mistral_api_key = 'apyBSpr2TCJj8TwRpVQCaBU4vKtyTUTB'
known_genres = {
    'action': ['action'],
    'adventure': ['adventure'],
    'animation': ['animation', 'animated'],
    'biography': ['biography', 'bio'],
    'comedy': ['comedy', 'comedies'],
    'crime': ['crime'],
    'documentary': ['documentary', 'doc'],
    'drama': ['drama'],
    'family': ['family'],
    'fantasy': ['fantasy'],
    'history': ['history', 'historical'],
    'horror': ['horror'],
    'music': ['music', 'musical', 'musicals'],
    'mystery': ['mystery'],
    'romance': ['romance', 'romantic'],
    'sci-fi': ['sci-fi', 'science fiction', 'scifi', 'science-fiction'],
    'sport': ['sport', 'sports'],
    'thriller': ['thriller'],
    'war': ['war'],
    'western': ['western'],
    'unknown': ['unknown']
}

# Create a mapping from alias to canonical genre
alias_to_genre = {}
for genre, aliases in known_genres.items():
    for alias in aliases:
        alias_to_genre[alias.lower()] = genre

def extract_genres_from_string(genre_str):
    if not isinstance(genre_str, str):
        return ['unknown']
    # Lowercase and split by comma or slash or semicolon
    tokens = re.split(r'[,/;]', genre_str.lower())
    genres_found = set()
    for token in tokens:
        token = token.strip()
        if token in alias_to_genre:
            genres_found.add(alias_to_genre[token])
        else:
            # Try partial match for multi-word genres
            for alias, genre in alias_to_genre.items():
                if alias in token:
                    genres_found.add(genre)
    if not genres_found:
        return ['unknown']
    return list(genres_found)