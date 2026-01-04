import pandas as pd
import matplotlib.pyplot as plt
from q1 import process_mati_data

df = process_mati_data("mati.csv")
WINDOW=7

def fill_missing_dates(daily_volume_df):
    earliest_date = daily_volume_df['dates'].min()
    latest_date = daily_volume_df['dates'].max()

    complete_date_range = pd.date_range(start=earliest_date, end=latest_date, freq='D')
    all_dates_df = pd.DataFrame({'dates': complete_date_range})

    daily_volume_df['dates'] = daily_volume_df['dates'].astype('datetime64[ns]')
    all_dates_df['dates'] = all_dates_df['dates'].astype('datetime64[ns]')

    updated_df = pd.merge(all_dates_df, daily_volume_df, on='dates', how='left')
    updated_df['volume'] = updated_df['volume'].fillna(0).astype(int)

    new_dates_count = len(complete_date_range) - len(daily_volume_df)
    print(f"Number of missing dates added: {new_dates_count}")
    updated_df['dates'] = updated_df['dates'].dt.date

    return updated_df


def plot_daily_volume(daily_volume_df, day_min, day_max):

    daily_volume_subset = daily_volume_df.iloc[day_min - 1:day_max]

    figure, axes = plt.subplots(1, 1)
    axes.plot(daily_volume_subset.index, daily_volume_subset["volume"], color="red", alpha=1, label='Daily Volume') 
    axes.plot(daily_volume_subset.index, daily_volume_subset["rolling_mean"], color="green", linewidth=0.5, label='7-Day Rolling Avg')
    axes.plot(daily_volume_subset.index, daily_volume_subset["threshold"], color="blue", linewidth=0.5, label='Threshold')
    axes.set_xlabel('Days after The Fire', color='black')
    axes.set_ylabel('Daily Volume', color='black')
    axes.set_title("Daily Tweet Volume", color='black')
    axes.grid(color='blue', linestyle='--', linewidth=0.5, alpha=0.7)
    axes.tick_params(axis='x', colors='black')
    axes.tick_params(axis='y', colors='black')

    axes.legend()
    plt.tight_layout()
    plt.savefig('q2_1st_2nd_3rd.png')
    plt.show()

def plot_burst(daily_volume_df,day_min,day_max):

    daily_volume_subset = daily_volume_df.iloc[day_min - 1:day_max]
    bursts = daily_volume_df[daily_volume_df['burst']]

    figure, axes = plt.subplots(1, 1)
    axes.plot(daily_volume_subset.index, daily_volume_subset["volume"], color="red", alpha=1, label='Daily Volume') 
    axes.plot(daily_volume_subset.index, daily_volume_subset["threshold"], color="blue", linewidth=0.5, label='Threshold')
    
    plt.scatter(bursts.index, bursts['volume'], color='black', s=100, label='Detected Burst', zorder=5)
    plt.title('Detected Bursts after The Fire', fontsize=16)
    plt.xlabel('Date Volume')
    plt.ylabel('Daily Tweet Volume')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('q2_4rth-burst.png')
    plt.show()



df["timestamps"] = pd.to_datetime(df['created_at'])
df["dates"] = df["timestamps"].dt.date

daily_volume_df = df.groupby("dates").size().reset_index(name="volume")
daily_volume = daily_volume_df["volume"]

earliest_date = df["dates"].min()
latest_date = df["dates"].max()
days_period = (latest_date - earliest_date).days
non_zero_days = len(daily_volume)
daily_volume_df = fill_missing_dates(daily_volume_df)

day_min = 1
day_max = len(daily_volume_df)
daily_volume_df['rolling_mean'] = daily_volume_df['volume'].rolling(window=WINDOW, min_periods=1).mean()

#subq 1,subq 2,subq 3
daily_volume_df['rolling_std'] = daily_volume_df['volume'].rolling(window=WINDOW, min_periods=1).std()
daily_volume_df['threshold'] = daily_volume_df['rolling_mean']+(2*daily_volume_df['rolling_std'])
#plot_daily_volume(daily_volume_df, day_min, day_max)

#subq 4
daily_volume_df['burst']= daily_volume_df['threshold']<daily_volume_df['volume']
print (daily_volume_df)
print('The number of burst are: ',len(daily_volume_df[daily_volume_df['burst'] == True]))
plot_burst(daily_volume_df,day_min,day_max)