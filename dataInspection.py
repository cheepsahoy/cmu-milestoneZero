import pandas as pd
from utils.dataProvider import getDataPath

eventsData = getDataPath('events.csv.gz')
events = pd.read_csv(eventsData)

watches = events.loc[events["event_type"] == "watch"]
ratings = events.loc[events["event_type"] == "rating"]

ratedPairs = ratings[["user_id", "movie_id"]].drop_duplicates()

watchedNotRated = watches.merge(
    ratedPairs,
    on=["user_id", "movie_id"],
    how="left",
    indicator = True
)

watchedNotRated = watchedNotRated.loc[watchedNotRated['_merge'] == 'left_only']
print(watchedNotRated)