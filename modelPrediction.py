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


def modelPredict(userID: int, numberOfMovies: int) -> pd.DataFrame:
    """
    Desc: This function uses the supplied model suggest the top numberOfMovies ranked by the models prediction.

    Args:
        userID: the userID we are making suggestions to
        numberOfMovies: integer, supplies a number of movies
    Returns:
        pd.DataFrames from the movies dataset.
    """
    if userID not in users["user_id"].values:
        raise ValueError(f"Error: {userID} is not valid")

    if numberOfMovies <= 0:
        raise ValueError("Error: numberOfMovies must be positive")
    ## exclude movies the user has already seen
    userEvents = events.loc[
        (events["user_id"] == userID) & (events["event_type"].isin(["watch", "rating"]))
    ]
    seenSet = set(userEvents["movie_id"].dropna())

    unseenMovies = movies.loc[~movies["movie_id"].isin(seenSet)]
    popularMovieCount = 100 + numberOfMovies
    ## get 100+numberOfMovies most popular movies as candidates and grab their movie_ids. Note, this is a design decision to improve the weakness of the current model but it has a cost (e.g., what if our selected user really likes to watch only obscure movies? This approach will not serve them and thus this decision should be rejected under a condition of more confidence in the model)
    candidateMovies = unseenMovies.nlargest(popularMovieCount, "popularity").copy()

    ## establish inputs preperation
    candidateInputs = candidateMovies[["movie_id"]].copy()
    candidateInputs["user_id"] = str(userID)

    modelInputs = {
        "user_id": candidateInputs["user_id"].to_numpy(),
        "movie_id": candidateInputs["movie_id"].to_numpy(),
    }

    ## run model predictions, rank and return
    predict = suppliedModel.predict(modelInputs, verbose=0)

    candidateMovies["predicted_rating"] = predict.flatten()

    return candidateMovies.nlargest(numberOfMovies, "predicted_rating")
