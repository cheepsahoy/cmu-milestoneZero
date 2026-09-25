import pandas as pd
import tensorflow as tf

from utils.dataProvider import getDataPath
from utils.dataSplitter import splitDataForML

eventsData = getDataPath("events.csv.gz")
events = pd.read_csv(eventsData)
dummyTrainData, testData = splitDataForML(events, 0.8)

## Extract relevent data columns
testRatings = testData[["user_id", "movie_id", "rating"]].copy()
testRatings["user_id"] = testRatings["user_id"].astype(str)
testRatings["movie_id"] = testRatings["movie_id"].astype(str)
testRatings["rating"] = testRatings["rating"].astype("float32")

## Build the test dataset
testFeatures = {
    "user_id": tf.constant(testRatings["user_id"].tolist()),
    "movie_id": tf.constant(testRatings["movie_id"].tolist()),
}

testLabels = tf.constant(testRatings["rating"].to_numpy().reshape(-1, 1))

testDataset = tf.data.Dataset.from_tensor_slices((testFeatures, testLabels)).batch(128)

## Assign model and test
model = tf.keras.models.load_model(
    "models/collaborative_filter_v1_ADAM_8020Split.keras"
)

results = model.evaluate(testDataset, return_dict=True)
print(results)

## Compare against baseline
meanRating = dummyTrainData["rating"].mean()

baselineRMSE = (((testData["rating"] - meanRating) ** 2).mean()) ** 0.5

print("Baseline RMSE:", baselineRMSE)

## comparing seen vs unseen movies:
knownMovies = set(dummyTrainData["movie_id"].dropna())

unseenMovies = ~testData["movie_id"].isin(knownMovies)

print("Total test ratings:", len(testData))
print("Ratings for unseen movies:", unseenMovies.sum())
print("Percentage unseen:", unseenMovies.mean() * 100)

print(
    "Average training ratings per movie:",
    dummyTrainData.groupby("movie_id").size().mean(),
)

## compare predictions with actual ratings
sample = testRatings.sample(n=50, random_state=123).copy()

predictions = model.predict(
    {
        "user_id": tf.constant(sample["user_id"].tolist()),
        "movie_id": tf.constant(sample["movie_id"].tolist()),
    },
    verbose=0,
)
sample["predicted_rating"] = predictions.flatten()

sample["absolute_error"] = (sample["rating"] - sample["predicted_rating"]).abs()

print(sample.round(2).to_string(index=False))
