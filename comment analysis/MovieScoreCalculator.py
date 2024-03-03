import pandas as pd

class MovieScoreCalculator:
    def __init__(self, movie_df_path: str):
        """
        Initialize the MovieScoreCalculator class with the path to the movie DataFrame CSV file.

        Args:
            movie_df_path (str): The path to the movie DataFrame CSV file.
        """
        self.movie_df = pd.read_csv(movie_df_path, on_bad_lines='skip')

    @staticmethod
    def convert_to_number(string: str) -> float:
        """
        Convert a string representation of a number with suffixes (K for thousands, M for millions) to a float.

        Args:
            string (str): The string representation of the number, possibly with a suffix.

        Returns:
            float: The converted number.
        """
        multiplier = {'M': 1_000_000, 'K': 1_000}
        if string[-1] in multiplier:
            return float(string[:-1]) * multiplier[string[-1]]
        else:
            return float(string)

    @staticmethod
    def convert_fraction_to_float(fraction_string):
        """
        Convert a string fraction in the format 'x/y' to a float.

        Args:
            fraction_string (str): The string representing the fraction.

        Returns:
            float: The float value of the fraction.
        """
        parts = fraction_string.split('/')
        numerator = float(parts[0])
        denominator = float(parts[1])
        return numerator / denominator

    @staticmethod
    def rescale_column(df: pd.DataFrame, input_column_name: str, output_column_name: str) -> pd.DataFrame:
        """
        Rescale a column in a DataFrame to have values between 0 and 10.

        Args:
            df (pd.DataFrame): The input DataFrame.
            input_column_name (str): The name of the input column to rescale.
            output_column_name (str): The name of the output column to store the rescaled values.

        Returns:
            pd.DataFrame: The DataFrame with the rescaled column added.
        """
        min_rating = df[input_column_name].min()
        max_rating = df[input_column_name].max()
        df[output_column_name] = ((df[input_column_name] - min_rating) / (max_rating - min_rating)) * 10
        return df

    def data_process(self) -> pd.DataFrame:
        """
        Preprocess the movie DataFrame by performing various operations.

        Returns:
            pd.DataFrame: The preprocessed DataFrame with selected columns.
        """
        df = self.movie_df.dropna()
        df.loc[:, 'Movie Total Rating'] = df['Movie Total Rating'].apply(self.convert_to_number)
        df.loc[:, 'Comment Rating'] = df['Comment Rating'].apply(self.convert_fraction_to_float)
        df.loc[:, 'Content'] = df['Content'].map({'positive': 1, 'negative': -2, 'neutral': 0})
        df.loc[:, 'Comment Rating'] = df['Comment Rating'].replace(0.0, 0.1)
        df.loc[:, 'Comment Score'] = df['Comment Rating'] * df['Content']
        aggregations = {
            'Movie Rating': 'first',
            'Movie Total Rating': 'first',
            'Total Reviews': 'first',
            'Comment Rating': 'first',
            'Content': 'first',
            'Comment Score': 'mean'
        }
        df = df.groupby(['Found Title']).agg(aggregations).reset_index()
        df['Rescaled Movie Total Rating'] = df['Movie Total Rating'] ** 0.2
        df = self.rescale_column(df, 'Rescaled Movie Total Rating', 'Rescaled Movie Total Rating Normal')
        df = self.rescale_column(df, 'Comment Score', 'Comment Score Normal')
        df['Movie Score'] = df['Movie Rating'] + df['Rescaled Movie Total Rating Normal'] + df['Comment Score Normal']
        df.rename(columns={'Found Title': 'Movie Title'}, inplace=True)
        return df[['Movie Title', 'Movie Score']]

    @staticmethod
    def movie_ordering(list_of_movies: list, df: pd.DataFrame) -> list:
        """
        Order a list of movies based on their scores in a DataFrame.

        Args:
            list_of_movies (list): A list of movie titles.
            df (pd.DataFrame): DataFrame containing movie data.

        Returns:
            list: A list of movie titles ordered by their scores.
        """
        selected_df = df[df['Movie Title'].isin(list_of_movies)]
        sorted_df = selected_df.sort_values(by='Movie Score', ascending=False)
        sorted_movie_names = sorted_df['Movie Title'].tolist()
        return sorted_movie_names

if __name__ == '__main__':
    test_movies = ['The Avengers', 'The Martyred Presidents', 'The Wolf of Wall Street', 'The Wizard of Oz', 'Madame X']

    movie_score_calculator = MovieScoreCalculator('reviews-sen english complete.csv')
    movie_score_df = movie_score_calculator.data_process()
    ordered_movies = movie_score_calculator.movie_ordering(test_movies, movie_score_df)
    print(ordered_movies)
