mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download jrobischon/wikipedia-movie-plots
kaggle datasets download mohammad26845/persian-movie-dataset-english-persian
unzip wikipedia-movie-plots.zip
rm wikipedia-movie-plots.zip
mv wiki_movie_plots_deduped.csv english.csv
unzip persian-movie-dataset-english-persian.zip
rm persian-movie-dataset-english-persian.zip
mv dataset.csv persian.csv
rm dataset.xlsx