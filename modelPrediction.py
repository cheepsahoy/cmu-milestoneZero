import pandas as pd
import tensorflow as tf

from utils.dataProvider import getDataPath

eventsData = getDataPath("events.csv.gz")
moviesData = getDataPath("movies.csv.gz")
usersData = getDataPath("users.csv.gz")

events = pd.read_csv(eventsData)
movies = pd.read_csv(moviesData)
users = pd.read_csv(usersData)

suppliedModel = tf.keras.models.load_model(
    "models/collaborative_filter_v1_ADAM_8020Split.keras"
)


def modelPredict(userID: int, numberOfMovies: int) -> list[pd.DataFrame]:
    """
    Desc: This function uses the supplied model suggest the top numberOfMovies ranked by the models prediction.

    Args:
        userID: the userID we are making suggestions to
        numberOfMovies: integer, supplies a number of movies
    Returns:
        List: of pd.DataFrames from the movies dataset.
    """
    if userID not in users["user_id"].values:
        raise ValueError(f"Error: {userID} is not valid")

    ## exclude movies the user has already seen
    seenMoves = set(events.loc(events["user_id"] == userID), "move_id")

    ## get 100+numberOfMovies most popular movies as candidates and grab their movie_ids
    popularMovieCount = 100 + numberOfMovies

    ## run model on those movies

    ## return numberOfMovies as movies pd.Dataframe
