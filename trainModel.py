import pandas as pd
import tensorflow as tf

from utils.dataProvider import getDataPath
from utils.dataSplitter import splitDataForML

eventsData = getDataPath("events.csv.gz")
events = pd.read_csv(eventsData)
trainData, _ = splitDataForML(events, 0.8)

## Extract relevent data columns
trainingRatings = trainData[["user_id", "movie_id", "rating"]].copy()
trainingRatings["user_id"] = trainingRatings["user_id"].astype(str)
trainingRatings["movie_id"] = trainingRatings["movie_id"].astype(str)
trainingRatings["rating"] = trainingRatings["rating"].astype("float32")

## feature extraction
uniqueUsers = trainingRatings["user_id"].unique()
uniqueMovies = trainingRatings["movie_id"].unique()

## building Tensorflow layers
userLookup = tf.keras.layers.StringLookup(
    vocabulary=uniqueUsers,
    mask_token=None,
)
movieLookup = tf.keras.layers.StringLookup(
    vocabulary=uniqueMovies,
    mask_token=None,
)

## building embeddings
embeddingSize = 32
userEmbeddings = tf.keras.layers.Embedding(
    input_dim=len(userLookup.get_vocabulary()),
    output_dim=embeddingSize,
)
movieEmbeddings = tf.keras.layers.Embedding(
    input_dim=len(movieLookup.get_vocabulary()),
    output_dim=embeddingSize,
)


sampleUser = tf.constant(["42"])
sampleMovie = tf.constant(["10+things+i+hate+about+you+1999"])

userIndex = userLookup(sampleUser)
movieIndex = movieLookup(sampleMovie)

userVector = userEmbeddings(userIndex)
movieVector = movieEmbeddings(movieIndex)

interaction = tf.reduce_sum(
    userVector * movieVector,
    axis=1,
)

print(userLookup)
print(userVector)
print(movieVector)
print(interaction)
