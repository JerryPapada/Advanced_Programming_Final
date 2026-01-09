import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from q1 import process_mati_data

os.makedirs("plots", exist_ok=True)

def analyze_weekly_activity():
    input_file = "data/mati_q1_relevant.csv"
    if os.path.exists(input_file):
        df = pd.read_csv(input_file)
    else:
        print("Processed file not found. Running Q1 preprocessing...")
        df = process_mati_data("mati.csv")

    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    df = df.dropna(subset=['created_at'])
    df['week_of_year'] = df['created_at'].dt.isocalendar().week
    df['day_of_week'] = df['created_at'].dt.day_name()

    # Rows = Weeks, Columns = Days, Values = Count of Tweets
    weekly_matrix = df.groupby(['week_of_year', 'day_of_week']).size().unstack(fill_value=0)

    # sort by chronological order.
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Reindex checks if all days exist; if a day has 0 tweets across all weeks, fill_value=0 adds it.
    weekly_matrix = weekly_matrix.reindex(columns=days_order, fill_value=0)
    weekly_matrix = weekly_matrix.sort_index()

    # Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(weekly_matrix, cmap='YlOrRd', linewidths=.5, annot=False)
    
    plt.title('Tweet Volume by Week and Day', fontsize=15)
    plt.xlabel('Day of the Week', fontsize=12)
    plt.ylabel('Week of the Year', fontsize=12)
    output_img = "plots/weekly_activity_heatmap.png"
    plt.savefig(output_img)
    plt.show()

if __name__ == "__main__":
    analyze_weekly_activity()