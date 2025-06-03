import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from colorama import init, Fore
import time
import sys

# Initialize colorama
init(autoreset=True)

# Load and preprocess the dataset
def load_data(file_path='imdb_top_1000.csv'):
    try:
        df = pd.read_csv(file_path)
        df['combined_features'] = df['Genre'].fillna('') + ' ' + df['Overview'].fillna('')
        return df
    except FileNotFoundError:
        print(Fore.RED + f"Error: The file '{file_path}' was not found.")
        exit()

movies_df = load_data()

# Vectorize the combined features and compute cosine similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['combined_features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# List all unique genres
def list_genres(df):
    return sorted(set(genre.strip() for sublist in df['Genre'].dropna().str.split(', ') for genre in sublist))

genres = list_genres(movies_df)

# Recommend movies based on filters (genre, mood, rating)
def recommend_movies(genres=None, mood=None, rating=None, top_n=5):
    filtered_df = movies_df
    if genres:
        filtered_df[filtered_df['Genre'].str.contains(genres, case=False, na=False)]
    if rating:
        filtered_df[filtered_df['IMDB_Rating'] >= rating]

    filtered_df = filtered_df.sample(frac=1).reset_index(drop=True)

    recommendations = []
    for inx, row in filtered_df.iterrows():
        overview = row['Overview']
        if pd.isna(overview):
            continue
        polarity = TextBlob(overview).sentiment.polarity 
        if (mood and ((TextBlob(mood).sentiment.polarity < 0 and polarity > 0) or polarity >= 0)) or not mood:
                    recommendations.append((row['Series_Title'], polarity))
        if len(recommendations) == top_n:
             break
        
    return recommendations if recommendations else "No suitable movie recommendations were found."


# Display recommendations
def display_recommendations(recs, name):
    print(Fore.YELLOW + f"\n AI-Analyzed Movie Recommendations for {name}")
    for idx, (title, polarity) in enumerate(recs, 1):
          sentiment = "Positive 😊" if polarity > 0 else "Negative 😞" if polarity < 0 else "Neutral 😐"
          print(f"{Fore.CYAN}{idx}.{title}(Polarity: {polarity:.2f}, {sentiment})")


# Small processing animation
def processing_animation():
    for _ in range(3):
        print(Fore.YELLOW + ".", end="", flush=True)
        time.sleep(0.5)

# Handle AI recommendation flow 🔍
def handle_ai(name):
    print(Fore.BLUE + "\nLet's find the perfect movie for you!")

    #show genres
    print(Fore.GREEN + "Available Genres: ", end="")
    for idx, genre in enumerate(genres, 1):
        print(f"{Fore.CYAN} {idx}. {genre}")
    print()

    while True:
        genre_input = input(Fore.YELLOW + "Enter genre number or name to start: ").strip()
        if genre_input.isdigit() and 1 <= int(genre_input) <= len(genres):
            genre = genres[int(genre_input) -1 ]
            break
        elif genre_input.title() in genres:
            genre = genre_input.title()
            break
    print(Fore.RED + "Invalid input. Try again.\n")
     
    mood = input(Fore.YELLOW + "How do you feel today? ").strip()


    # Processing animation while analyzing mood 
    print(Fore.BLUE + "Analyzing mood", end="", flush=True)
    processing_animation()
    polarity = TextBlob(mood).sentiment.polarity
    mood_desc = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
    print(f"\n{Fore.GREEN} Your mood is {mood_desc} (Polarity: {polarity:.2f})")

    while True:
        rating_input = input(Fore.YELLOW + "Enter a minimum IMDB rating (7.6 - 9.3) or 'skip' to skip this step: ").strip()
        if rating == None:
            break
        try:
            rating = float(rating_input)
            if 7.6 <= rating <=9.3:
                break
            print(Fore.RED + "Rating out of range. Try again.\n")
        except ValueError:
            print(Fore.RED + "Invalid input. Try again.\n")
    

    # Processing animation while finding movies
    print(f"{Fore.BLUE}\nFinding movies for {name}", end="", flush=True)
    processing_animation()


    # Small processing animation while finding movies 
    recs = recommend_movies(genre=genre, mood=mood, rating=rating, top_n=5)
    if isinstance(recs, str):
        print(Fore.RED + recs + "\n")
    else:
        display_recommendations(recs, name)
    while True:
        action = input(Fore.YELLOW + "\nWould you like more movie recommendations? (Y/N): ").strip().lower()
        if action == 'no':
            print(Fore.GREEN + (f"Enjoy your movie picks, {name}!"))
            break
        elif action == 'yes':
            recs = recommend_movies(genre=genre, mood=mood, rating=rating, top_n=5)
            if isinstance(recs, str):
                print(Fore.RED + recs + "\n")
            else:
                display_recommendations(recs, name)
        else:
            print(Fore.RED + "Invalid choice. Try again.\n")
    

# Main program 
def main():
    print(Fore.BLUE + "Welcome to your Personal Movie Recommendation Assistant!\n")
    name = input(Fore.YELLOW + "What is your name? ").strip()

    print(f"\n{Fore.GREEN}Great to meet you, {name}!\n")
    handle_ai(name)

if __name__ == "__main__:":
    main()