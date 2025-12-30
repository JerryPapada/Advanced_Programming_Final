# This script file provides fundamental processsing operations on the Tempi data
# that were collected by utilizing the Streaming API of twitter.

# Import the required Python libraries.
import pandas as pd
import matplotlib.pyplot as plt
import string
from tqdm import tqdm
from collections import Counter
import spacy

# Use the following terminal command to download the greek corpus of spacy.
# python -m spacy download el_core_news_sm

# Load the Greek language model
nlp = spacy.load("el_core_news_sm")

# =============================================================================
#                         Functions Definition:
# =============================================================================
def plot_daily_volume(daily_volume_df, day_min, day_max):
    # This function generates a plot of daily tweet volumes for a specified 
    # period, with validation.
    
    # Input Parameters:
    # - daily_volume_df (DataFrame): A DataFrame containing a 'date' column 
    #                                (timestamps of dates)  and a 'volume' 
    #                                column (corresponding daily volume).
    # - day_min (int): Starting day (1-indexed).
    # - day_max (int): Ending day (1-indexed).
    
    # The function ensures the provided day range is valid and plots the daily 
    # tweet volumes for the specified range.
    
    # Validate day_min and day_max
    total_days = len(daily_volume_df)
    if day_min < 1 or day_max > total_days or day_min > day_max:
        raise ValueError(
            f"Invalid day range: day_min={day_min}, day_max={day_max}. "
            f"Valid range is 1 to {total_days}, and day_min should not exceed day_max."
        )

    # Extract the subset of data based on the day range
    daily_volume_subset = daily_volume_df.iloc[day_min - 1:day_max]

    # Plot the daily volume of tweets since day zero
    figure, axes = plt.subplots(1, 1)
    
    # Set a dark background
    plt.style.use('dark_background')
    
    # Plot the data
    axes.plot(daily_volume_subset["volume"], color="red", alpha=0.8)
    
    # Add axes labels
    axes.set_xlabel('Time Since Accident', color='white')
    axes.set_ylabel('Daily Volume', color='white')
    
    # Add title to the figure
    axes.set_title("Daily Evolution of Tweets", color='white')
    
    # Add grid
    axes.grid(color='yellow', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Customize tick colors for dark background
    axes.tick_params(axis='x', colors='white')
    axes.tick_params(axis='y', colors='white')
    
    # Show the plot
    plt.show()
# =============================================================================
def plot_author_volume(tweets_df,day_min,day_max):
    # This function computes the volume of tweets per author for a given range 
    # of days and generates the corresponding histogram plot.
    
    # Input Parameters:
    # - tweets_df (DataFrame): A DataFrame containing:
    #                          - 'author_id': ID of the author.
    #                          - 'dates': Dates of the tweets (datetime format).
    # - day_min (int): The starting day (1-indexed) from the earliest date.
    # - day_max (int): The ending day (inclusive, 1-indexed).
    
    # Output Parameters:
    # - result_df (DataFrame): A DataFrame with columns:
    #                          - 'author_id': The unique author IDs.
    #                          - 'volume': The number of tweets per author in 
    #                             the given range.
    
    # Get the earliest and latest dates
    earliest_date = tweets_df["dates"].min()
    latest_date = tweets_df["dates"].max()
    
    # Total number of days in the dataset
    total_days = (latest_date - earliest_date).days + 1
    
    # Validate the specified range
    if day_min < 1 or day_max > total_days or day_min > day_max:
        raise ValueError(
            f"Invalid day range: day_min={day_min}, day_max={day_max}. "
            f"Valid range is from 1 to {total_days} days."
        )
    
    # Calculate the date range corresponding to day_min and day_max
    start_date = earliest_date + pd.Timedelta(days=day_min - 1)
    end_date = earliest_date + pd.Timedelta(days=day_max - 1)
    
    # Filter the dataset for the specified range of days
    filtered_data = tweets_df[(tweets_df["dates"] >= start_date) & 
                            (tweets_df["dates"] <= end_date)]
    
    # Group by author_id and count the tweets
    author_volume_df = filtered_data.groupby("author_id").size().reset_index(name="volume")
    
    # Sort by tweet volume in descending order
    author_volume_df = author_volume_df.sort_values(by="volume", ascending=False).reset_index(drop=True)
    
    # Isolate the volume of tweets per author
    author_volume = author_volume_df["volume"]

    # Generate the corresponding author volume range list.
    author_volume_bins = [v for v in range(min(author_volume),max(author_volume)+1)]

    # Set the largest k volumes of tweets to be represented in the frequency histogram
    low_k_volumes = 60
    # Set up a figure with a set of axes
    figure, axes = plt.subplots(1, 1)
    # Set a dark background
    plt.style.use('dark_background')
    # Generate the histogram
    axes.hist(author_volume, bins=author_volume_bins[:low_k_volumes], color="red", alpha=0.7)
    # Add axes labels
    axes.set_xlabel('Tweets Volume', color='white')
    axes.set_ylabel('Number of Authors', color='white')
    # Set the title string.
    title_string = f"Distribution of Authors per Number of Tweets from Day {day_min} to Day {day_max}"
    # Add title to the figure
    axes.set_title(title_string, color='white')
    # Add grid
    axes.grid(color='yellow', linestyle='--', linewidth=0.5, alpha=0.7)
    # Customize tick colors for dark background
    axes.tick_params(axis='x', colors='white')
    axes.tick_params(axis='y', colors='white')
    # Show the plot
    plt.show()
    
    return author_volume_df
# =============================================================================
def fill_missing_dates(daily_volume_df):
    # This function identifies missing dates in the dataset, adds them with 
    # zero volume, and reports the number of new dates added.

    # Input Parameters:
    # - daily_volume_df (DataFrame): A DataFrame containing:
    #                                - 'date': A column of timestamps 
    #                                          representing dates.
    #                                - 'volume': A column of volumes 
    #                                            corresponding to each date.

    # Output Parameters:
    # - updated_df (DataFrame): The updated DataFrame with all missing dates 
    #                           filled with zero volume.
    
    # Determine the range of dates
    earliest_date = daily_volume_df['dates'].min()
    latest_date = daily_volume_df['dates'].max()

    # Generate a complete date range from earliest to latest date
    complete_date_range = pd.date_range(start=earliest_date, end=latest_date, freq='D')

    # Create a DataFrame for the complete date range
    all_dates_df = pd.DataFrame({'dates': complete_date_range})

    # Merge with the existing DataFrame to identify missing dates
    # Ensure both columns are of the same type (datetime64[ns])
    daily_volume_df['dates'] = daily_volume_df['dates'].astype('datetime64[ns]')
    all_dates_df['dates'] = all_dates_df['dates'].astype('datetime64[ns]')

    updated_df = pd.merge(all_dates_df, daily_volume_df, on='dates', how='left')

    # Fill missing volumes with zero
    updated_df['volume'] = updated_df['volume'].fillna(0).astype(int)

    # Report the number of missing dates added
    new_dates_count = len(complete_date_range) - len(daily_volume_df)
    print(f"Number of missing dates added: {new_dates_count}")

    # Ensure the 'dates' column contains only the date component
    updated_df['dates'] = updated_df['dates'].dt.date

    return updated_df
# =============================================================================
def preprocess_text(text):
    # This function preprocesses a Pandas Series of text data by performing the 
    # following operations:
    # 1. Converts text to lowercase.
    # 2. Removes mentions (e.g., @username).
    # 3. Removes retweet indicators (e.g., 'RT').
    # 4. Removes links (e.g., http://example.com, www.example.com).
    # 5. Removes only the hashtag mark (#) but keeps the text following it.
    # 6. Replaces words containing punctuation with their non-punctuated versions.
    # 7. Removes ellipsis-like characters (e.g., …).
    # 8. Removes consecutive punctuation marks (e.g., !!!, ??).
    # 9. Removes single punctuation characters such as commas, colons, and hyphens.
    # 10. Removes emoticons and other symbols using Unicode ranges.
    # 11. Removes words with less than 4 characters.
    # 12. Collapses multiple spaces into one and trims leading or trailing spaces.
    
    # Input Parameters:
    # - text (pd.Series): Pandas Series containing the text to preprocess.
    
    # Output Parameters:
    # - pd.Series: Cleaned text.
    
    def remove_punctuation_from_words(line):
        # Splits the line into words and removes punctuation from each word
        words = line.split()
        # Remove punctuation from each word
        cleaned_words = [word.translate(str.maketrans("", "", string.punctuation)) for word in words]
        # Join the cleaned words back into a single string
        return " ".join(cleaned_words)
    
    # Convert text to lowercase
    text = text.str.lower()    
    # Remove mentions (@username)
    text = text.str.replace(r"@\w+", "", regex=True)  # Matches any word starting with '@'
    # Remove retweet indicators (e.g., RT)
    text = text.str.replace(r"\b(rt)\b", "", regex=True)  # Matches 'rt' as a whole word
    # Remove links (e.g., http://example.com or www.example.com)
    text = text.str.replace(r"http\S+|www\S+", "", regex=True)  # Matches URLs starting with http or www
    # Remove only the hashtag mark (#) but keep the text after it
    text = text.str.replace(r"#", "", regex=True)  # Matches the '#' character
    # Replace words containing punctuation with their non-punctuated version
    text = text.apply(remove_punctuation_from_words)  # Applies word-level punctuation removal
    # Remove ellipsis-like characters (…)
    text = text.str.replace(r"…", "", regex=True)  # Matches the Unicode ellipsis character '…'
    # Remove consecutive punctuation marks (e.g., !!!, ??)
    text = text.str.replace(r"[.!?;&]{1,}", "", regex=True)  # Matches one or more consecutive ., !, ;,& or ?
    # Remove single full stops, commas, colons, and hyphens
    text = text.str.replace(r"[.,:\-]", "", regex=True)  # Matches ., :, -, or ,
    # Remove emoticons using Unicode ranges
    text = text.str.replace(
        r"[\U0001F600-\U0001F64F" # Emoticons
        r"\U0001F300-\U0001F5FF"  # Miscellaneous Symbols and Pictographs
        r"\U0001F680-\U0001F6FF"  # Transport and Map Symbols
        r"\U0001F1E0-\U0001F1FF"  # Flags (iOS)
        r"\U00002600-\U000026FF"  # Miscellaneous Symbols
        r"\U00002700-\U000027BF"  # Dingbats
        r"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        r"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        r"]+", 
        "", 
        regex=True
    )  # Matches Unicode ranges for various emoji and symbols
    # Remove words with less than 4 characters
    text = text.str.replace(r'\b\w{1,3}\b', '', regex=True)  # Matches whole words with length <= 3
    # Remove extra spaces caused by word removal
    text = text.str.replace(r'\s+', ' ', regex=True).str.strip()  # Collapses multiple spaces into one and trims
    return text
# =============================================================================  
def clean_tweets(texts):
    # This function cleans tweets in bulk using spaCy's nlp.pipe() for efficient 
    # processing.
    
    # Input Parameters:
    # - texts (list of str): List of text entries to clean.
    
    # Output Parameters:
    # - pd.Series: A pandas Series containing the cleaned text.
    
    cleaned_texts = []
    for doc in tqdm(nlp.pipe(texts, batch_size=1000, disable=["parser", "ner", "tagger"]), 
                    total=len(texts), desc="Cleaning tweets"):
        # Process tokens: retain punctuation, remove stopwords, non-alphanumeric tokens, and short words
        tokens = [
            token.text for token in doc
            if not token.is_stop and token.is_alpha and len(token.text) >= 4
        ]
        cleaned_texts.append(" ".join(tokens))
    
    # Return the cleaned texts as a pandas Series
    return pd.Series(cleaned_texts)
# =============================================================================
def plot_top_k_terms_histogram(text, k=10):
    # This function Generates a histogram of the top k most frequent terms in 
    # the preprocessed text of the tweets dataframe.
    
    # Input Parameters:
    # - text : Series object containing the preprocessed tweet content.
    # - k (int): Number of most frequent terms to display in the histogram.
    
    # Output Parameters:
    # - term_frequencies (dict): A dictionary where keys are terms and values 
    #                            are their frequencies.
    
    # Tokenize the text entries and flatten into a single list of words
    all_words = []
    text.dropna().apply(lambda x: all_words.extend(x.split()))
    
    # Count word frequencies
    word_counts = Counter(all_words)
    
    # Get the top k most frequent terms
    top_k_terms = word_counts.most_common(k)
    terms, frequencies = zip(*top_k_terms)
    

    # Set up a figure with a set of axes
    figure, axes = plt.subplots(1, 1, figsize=(10, 6))
    # Set a dark background
    plt.style.use('dark_background')
    # Generate the bar plot
    axes.bar(terms, frequencies, color='red', alpha=0.7)
    # Add axes labels
    axes.set_xlabel('Terms', color='white')
    axes.set_ylabel('Frequency', color='white')
    # Set the title string
    title_string = f"Top {k} Most Frequent Terms in Tweets"
    axes.set_title(title_string, color='white')
    # Add grid
    axes.grid(color='yellow', linestyle='--', linewidth=0.5, alpha=0.7)
    # Customize tick colors for the dark background
    axes.tick_params(axis='x', colors='white', labelrotation=45, labelsize=10)
    axes.tick_params(axis='y', colors='white')
    # Adjust layout to prevent clipping
    figure.tight_layout()
    # Show the plot
    plt.show()
    
    # Return the dictionary of term frequencies
    term_frequencies = dict(word_counts)
    return term_frequencies
# =============================================================================

# Set the csv file name containing the acquired data either to mati.csv or
# to tempi.csv
filename = 'tempi.csv'

# Read the file into a pandas dataframe.
tweets_df = pd.read_csv(filename)
# Set names of the series objects that form the given dataframe.
tweets_df.columns = ["author_id","created_at","geo","tweet_id","lang","like_count",
                    "quote_count","reply_count","retweet_count","source","text"]
# Incorporate an additional column to store the created_at strings as 
# timestamp objects.
tweets_df["timestamps"] = pd.to_datetime(tweets_df['created_at'])
# Incorporate an additional column to store the extracted dates from the 
# corresponding timestamps.
tweets_df["dates"] = tweets_df["timestamps"].dt.date

# Compute the daily volume of tweets.
daily_volume_df = tweets_df.groupby("dates").size().reset_index(name="volume")
# Isolate the daily volume of tweets.
daily_volume = daily_volume_df["volume"]

# Calculate the time period in days that spans the dataset.
earliest_date = tweets_df["dates"].min()
latest_date = tweets_df["dates"].max()
days_period = (latest_date - earliest_date).days
# Calculate the number of dates in the dataset with non-zero volume of tweets.
non_zero_days = len(daily_volume)

# Report the earliest and latest dates in the dataset alonside with the total
# time duration in days.
print(f"Earliest date: {earliest_date}")
print(f"Latest date: {latest_date}")
print(f"Time period in days: {days_period}")
print(f"Number of non-zero volume days: {non_zero_days}")

# Update the daily volume dataframe to incorporate the missing dates with zero
# volume entries.
daily_volume_df = fill_missing_dates(daily_volume_df)

# Plot the daily volume of tweets since day zero for the complete time range of
# the dataset measured in days.
day_min = 1
day_max = len(daily_volume_df)
plot_daily_volume(daily_volume_df, day_min, day_max)

# Get the volume of tweets per unique author for the complete time range of the
# dataset measured in days.
author_volume_df = plot_author_volume(tweets_df, day_min, day_max)

# Report the total number of tweets and the corresponding total number of unique
# authors.
total_tweets = len(tweets_df)
unique_authors = len(author_volume_df)
print(f"{unique_authors} unique authors tweeted {total_tweets} tweets in total")

# Generate a new dataframe called author_volume_time_span which for each entry 
# will store the author id, the author volume and the time span in days between 
# the first and last tweet of every single author.

# Group by 'author_id' and compute the required metrics
author_volume_time_span = (
    tweets_df.groupby('author_id')['dates']
    .agg(
        first_tweet_date='min',   # First tweet date
        last_tweet_date='max',    # Last tweet date
        tweet_volume='count'      # Number of tweets
    )
    .reset_index()
)

# Ensure first_tweet_date and last_tweet_date are datetime
author_volume_time_span['first_tweet_date'] = pd.to_datetime(author_volume_time_span['first_tweet_date'])
author_volume_time_span['last_tweet_date'] = pd.to_datetime(author_volume_time_span['last_tweet_date'])

# Compute the time span in days
author_volume_time_span['time_span_days'] = (
    (author_volume_time_span['last_tweet_date'] - author_volume_time_span['first_tweet_date']).dt.days
)

# Sort the author_volume_time_span dataframe according to the time span in 
# descending order.
author_volume_time_span = author_volume_time_span.sort_values(
    by='time_span_days', ascending=False).reset_index(drop=True)

# Identify the unique tweets.

# Sort by timestamps in ascending order
tweets_df = tweets_df.sort_values(by='timestamps', ascending=True)

# Apply the text purification process.
tweets_df["text"] = preprocess_text(tweets_df["text"])
# Clean text in bulk and incorporate into the DataFrame
# tweets_df["cleaned_text"] = clean_tweets(tweets_df["text"].tolist())

# Drop duplicates based on the 'text' column, keeping the first occurrence
unique_tweets_df = tweets_df.drop_duplicates(subset='text', keep='first').reset_index(drop=True)

# Generate the histogram of the k top most frequent terms in the unique tweets.
term_frequencies = plot_top_k_terms_histogram(unique_tweets_df["text"], k=20)

# Short dictionary in descending order of frequency.
term_frequencies = dict(
    sorted(term_frequencies.items(), key=lambda item: item[1], reverse=True)
)