import pandas as pd

from utils.dataProvider import getDataPath

moviesData = getDataPath("movies.csv.gz")
movies = pd.read_csv(moviesData)
