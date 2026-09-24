import pandas as pd


def splitDataForML(
    events: pd.DataFrame, splitPercent: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if splitPercent <= 0 or splitPercent >= 1:
        raise ValueError("splitPercent must be between (0, 1) exclusive")

    ## in the current dataset, # of events["event_type"].value_counts() yields equivlent # of ratings and watchthis logic will likely need to be edited later
    ratings = events.loc[events["event_type"] == "rating"].copy()

    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"])

    trainData = []
    testData = []

    ## go through events by userID, find all their rating cases
    for _, userRatings in ratings.groupby("user_id"):
        ## if only one rating, we want to make sure its in training data, not test
        if len(userRatings) == 1:
            trainData.append(userRatings)
            continue

        ## sort by timestamp
        userRatings = userRatings.sort_values("timestamp")

        ## split by splitPercent variable, clamped to ensure split
        splitIndex = max(
            1, min(int(len(userRatings) * splitPercent), len(userRatings) - 1)
        )

        trainData.append(userRatings.iloc[:splitIndex])
        testData.append(userRatings.iloc[splitIndex:])

    trainExport = pd.concat(trainData)
    testExport = pd.concat(testData)

    return (trainExport, testExport)
