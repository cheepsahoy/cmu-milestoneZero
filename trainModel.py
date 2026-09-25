import pandas as pd

from utils.dataProvider import getDataPath
from utils.dataSplitter import splitDataForML

eventsData = getDataPath("events.csv.gz")
moviesData = getDataPath("movies.csv.gz")
usersData = getDataPath("users.csv.gz")

events = pd.read_csv(eventsData)
movies = pd.read_csv(moviesData)
users = pd.read_csv(usersData)


trainData, _ = splitDataForML(events)

userIdUnique = users["user_id"].unique()
movieIdUnique = movies["move_id"].unique()
