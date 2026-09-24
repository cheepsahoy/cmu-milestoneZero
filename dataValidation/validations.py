## Validation Rules
ratingValidations = [
    "Rating vs Watching: We want to make sure that the rating happens AFTER the WATCHING to be relevent training data Initial data means there is not a single watch that completes without a rating -- in the future this might not be the case and we should register watch w/o rating as neutral if the movie finishes, and BAD if it doesn't"
]

## Data Types and Shapes
eventDataShape = {
    "timestamp": {"dtype": "object", "description": "event timestamp"},
    "user_id": {"dtype": "int64"},
    "event_type": {
        "dtype": "object",
        "allowed": ["watch", "rating", "account_created"],
    },
    "movie_id": {
        "dtype": "object",
        "description": "A combination of lowercase title and year seperated by +, e.g.: twelve+monkeys+1995",
        "nullable": True,
    },
    "rating": {
        "dtype": "object",
        "range": [1, 10, float("nan")],
    },
}

usersDataShape = {
    "user_id": {"dtype": "int64"},
    "age": {"dtype": "int64"},
    "occupation": {
        "dtype": "object",
        "allowed": [
            "college/grad student",
            "executive/managerial",
            "sales/marketing",
            "scientist",
            "other or not specified",
            "self-employed",
            "academic/educator",
            "artist",
            "retired",
            "K-12 student",
            "homemaker",
            "clerical/admin",
            "technician/engineer",
            "tradesman/craftsman",
            "programmer",
            "writer",
            "customer service",
            "unemployed",
            "lawyer",
            "doctor/health care",
        ],
        "gender": {"dtype": "obj", "allowed": ["M", "F"]},
        "self_description_likes": {"dtype": "obj", "nullable": True},
        "self_description_dislikes": {"dtype": "obj", "nullable": True},
    },
}

movieDataShape = {
    "movie_id": {
        "dtype": "object",
        "description": "A combination of lowercase title and year seperated by +, e.g.: twelve+monkeys+1995",
        "nullable": True,
    },
    "title": {"dtype": "object"},
    "genres": {
        "dtype": "object",
        "description": "genres are represented as upper-case first letter and split by |. E.g.,: Drama|Romance or Western|Action|Adventure",
    },
    "release_date": {"dtype": "object"},
    "runtime": {"dtype": "int64"},
    "origional_language": {"dtype": "object"},
    "overview": {"dtype": "object"},
    "popularity": {"dtype": "float64'"},
    "vote_average": {"dtype": "float64"},
    "vote_count": {"dtype": "int64"},
    "imdb_id": {"dtype": "object"},
    "tmdb_id": {"dtype": "int64"},
    "license_cost": {"dtype": "float64"},
}
