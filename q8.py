import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from q1 import process_mati_data


os.makedirs("plots", exist_ok=True)

def generate_three_diagrams():
    df = process_mati_data("mati.csv")
    
    # Identify Retweets (Text starts with "RT ")
    df['is_retweet'] = df['text'].astype(str).str.contains(r'^RT\s', regex=True, na=False)
    
    # Author Statistics
    author_stats = df.groupby('author_id').agg(
        total_tweets=('text', 'count'),
        retweet_count=('is_retweet', 'sum')
    ).reset_index()
    
    # Calculate Dependency %
    author_stats['retweet_percentage'] = (author_stats['retweet_count'] / author_stats['total_tweets']) * 100

    total_retweets = df['is_retweet'].sum()
    total_original = len(df) - total_retweets
    
    plt.figure(figsize=(7, 7))
    plt.pie(
        [total_retweets, total_original], 
        labels=[f'Retweets\n({total_retweets})', f'Original\n({total_original})'], 
        autopct='%1.1f%%', 
        colors=['#ff9999', '#66b3ff'], 
        startangle=140,
        explode=(0.05, 0)
    )
    plt.title('Global Percentage of Retweets in Dataset')
    plt.savefig('plots/diagram1_global_retweet_percent.png')
    plt.show()


    # This shows the RELATIVE distribution (Y-axis = %)
    plt.figure(figsize=(10, 6))
    sns.histplot(
        author_stats['retweet_percentage'], 
        bins=20, 
        kde=False, 
        stat="percent",
        color='mediumpurple'
    )
    plt.title('Author Retweet Dependency (% of Authors)')
    plt.xlabel('Retweet Percentage (0% = Original, 100% = Pure RT)')
    plt.ylabel('Percentage of Authors (%)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('plots/diagram2_author_dependency_percent.png')
    plt.show()

    # This shows the ABSOLUTE distribution (Y-axis = Count)
    plt.figure(figsize=(10, 6))
    sns.histplot(
        author_stats['retweet_percentage'], 
        bins=20, 
        kde=False, 
        stat="count",
        color='teal'
    )
    plt.title('Author Retweet Dependency')
    plt.xlabel('Retweet Percentage (0% = Original, 100% = Pure RT)')
    plt.ylabel('Number of Authors (Count)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('plots/diagram3_author_dependency_real_values.png')
    plt.show()

if __name__ == "__main__":
    generate_three_diagrams()