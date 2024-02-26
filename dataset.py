import pandas as pd

if __name__ == "__main__":
    wiki_movies = pd.read_csv("datasets/english.csv")
    persian_movies = pd.read_csv("datasets/persian.csv")
    persian_movies.drop(["Content_1", "PERSIAN_title", "PENGLISH_title", "Score", "Time", "Link"], axis=1, inplace=True)
    persian_movies["Origin/Ethnicity"] = "persian"
    persian_movies.rename(columns = {'EN_title':'Title', "Content_2":"Plot", "Year":"Release Year"}, inplace = True)
    persian_movies = persian_movies.reindex(sorted(persian_movies.columns), axis=1)
    wiki_movies.drop(["Wiki Page", "Cast", "Director"], axis=1, inplace=True)
    wiki_movies.rename(columns = {'EN_title':'Title', "Content_2":"Content"}, inplace = True)
    wiki_movies = wiki_movies.reindex(sorted(wiki_movies.columns), axis=1)
    all_movies = pd.concat([wiki_movies, persian_movies], ignore_index=True)
    all_movies.fillna("unknown", inplace=True)
    all_movies["Genre"] = all_movies["Genre"].apply(lambda x: "unknown" if x in ["", " ", "-", "[140]", "[144]"] else x)
    all_movies = all_movies[['Title', 'Release Year', 'Genre', 'Origin/Ethnicity', 'Plot']]
    all_movies.rename({'Origin/Ethnicity': 'Origin', 'Release Year': 'Year'}, axis=1, inplace=True)
    all_movies.to_csv("datasets/movies.csv", index=False)