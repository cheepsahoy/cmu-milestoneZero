import pandas as pd
from utils.dataProvider import getDataPath

eventsData = getDataPath('events.csv.gz')
moviesData = getDataPath('movies.csv.gz')
usersData = getDataPath('users.csv.gz')

events = pd.read_csv(eventsData)
movies = pd.read_csv(moviesData)
users = pd.read_csv(usersData)

print("EVENTS")
print(events.shape)
print(events.columns)
print(events.dtypes)
print(events["event_type"].value_counts())
print(events.head())
print(events.info())

print("\nUSERS")
print(users.shape)
print(users.columns)
print(users.dtypes)
print(users.head())
print(users.info())

print("\nMOVIES")
print(movies.shape)
print(movies.columns)
print(movies.dtypes)
print(movies.head())
print(movies.info())