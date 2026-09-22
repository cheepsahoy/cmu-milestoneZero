import pandas as pd
from utils.dataProvider import getDataPath

eventsData = getDataPath('events.csv.gz')
moviesData = getDataPath('movies.csv.gz')
usersData = getDataPath('users.csv.gz')

df = pd.read_csv(usersData)
print(df.head())