import pandas as pd

from utils.dataProvider import getDataPath
from utils.dataSplitter import splitDataForML

eventsData = getDataPath("events.csv.gz")
moviesData = getDataPath("movies.csv.gz")
usersData = getDataPath("users.csv.gz")

events = pd.read_csv(eventsData)
movies = pd.read_csv(moviesData)
users = pd.read_csv(usersData)

trainData, testData = splitDataForML(events, 0.8)

badSplits = []

for userID in testData["user_id"].unique():
    userTrain = trainData.loc[trainData["user_id"] == userID]

    userTest = testData.loc[testData["user_id"] == userID]

    latestTrain = userTrain["timestamp"].max()
    earliestTest = userTest["timestamp"].min()

    if latestTrain > earliestTest:
        badSplits.append(userID)

print("Users with bad chronological splits:", len(badSplits))
