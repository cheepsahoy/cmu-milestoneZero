## Milestone Zero

Movie recommendation algorithm for CMU ML project.

## Running the Project
Point of entry is `main.py`. You should use the provided `reccomendMetaFunction` and supply a `userID` of your choice and an intger `numberOfMovies`. The function will a pd.Dataset from the `movies.csv` file.


## Setting Up Your Own Environment!
Running is pretty straightforward, two things you need to take care of: Data, and API Tokens.

1) Data: Repository runs on the dataset provided for milestone 0. Not pushed to repository for "privacy" protection. If running on your own, you'll want to make a folder called 'data' at project root and place the `csv.gz` files in them. Names are: `events.csv`, `movies.csv`, and `users.csv`.

2) API Token: Project runs through instructor for typesafe responses. Configured arround open ai. To run on your own, you'll need to make a `.env` file and ensure `coldStart` is pointing at it. You'll want your .env file to have an `OPENAI_API_KEY` and `OPENAI_MODEL` variable. I reccomend (for cheapness) `gpt-4o-mini`.