from pathlib import Path

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

## grab unique users and movies
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

## build input
userInput = tf.keras.Input(shape=(), dtype=tf.string, name="user_id")
movieInput = tf.keras.Input(shape=(), dtype=tf.string, name="movie_id")

userIndex = userLookup(userInput)
movieIndex = movieLookup(movieInput)
userVector = userEmbeddings(userIndex)
movieVector = movieEmbeddings(movieIndex)

interaction = tf.keras.layers.Dot(axes=1, name="interaction")([userVector, movieVector])

## account for user biases
userBiasesLayer = tf.keras.layers.Embedding(
    input_dim=len(userLookup.get_vocabulary()),
    output_dim=1,
    name="user_bias",
)

movieBiasesLayer = tf.keras.layers.Embedding(
    input_dim=len(movieLookup.get_vocabulary()),
    output_dim=1,
    name="movie_bias",
)
userBias = userBiasesLayer(userIndex)
movieBias = movieBiasesLayer(movieIndex)

## construct model
combined = tf.keras.layers.Add()([interaction, userBias, movieBias])
meanRating = float(trainingRatings["rating"].mean())

prediction = tf.keras.layers.Rescaling(
    scale=1.0,
    offset=meanRating,
    name="predicted_rating",
)(combined)

model = tf.keras.Model(
    inputs={
        "user_id": userInput,
        "movie_id": movieInput,
    },
    outputs=prediction,
    name="collaborative_filter_v1",
)

## preparing dataset for model
features = {
    "user_id": tf.constant(trainingRatings["user_id"].tolist()),
    "movie_id": tf.constant(trainingRatings["movie_id"].tolist()),
}
labels = tf.constant(trainingRatings["rating"].to_numpy().reshape(-1, 1))

trainDataset = tf.data.Dataset.from_tensor_slices((features, labels))
trainDataset = trainDataset.shuffle(len(trainingRatings), seed=12345).batch(128)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
)

## train the model
history = model.fit(trainDataset, epochs=10)

## save the model
Path("models").mkdir(exist_ok=True)
model.save("collaborative_filter_v1_ADAM_8020Split.keras")
