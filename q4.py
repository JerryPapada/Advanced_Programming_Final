import pandas as pd
import matplotlib.pyplot as plt
from q1 import process_mati_data

df = process_mati_data("mati.csv")

df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
df = df.dropna(subset=['created_at'])
df['hour'] = df['created_at'].dt.hour
df['date'] = df['created_at'].dt.date

# average
hourly_counts = df.groupby(['date', 'hour']).size().reset_index(name='count')
avg_activity = hourly_counts.groupby('hour')['count'].mean().reset_index()

# fill missing hours ()
all_hours = pd.DataFrame({'hour': range(24)})
avg_activity = pd.merge(all_hours, avg_activity, on='hour', how='left').fillna(0)
avg_activity = avg_activity.sort_values('hour')

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(avg_activity['hour'], avg_activity['count'], color='red', alpha=0.7)
ax.set_title('Average Tweet Volume by Hour of Day', color='white')
ax.set_xlabel('Hour of Day', color='white')
ax.set_ylabel('Average Number of Tweets', color='white')
ax.grid(color='yellow', linestyle='--', linewidth=0.5, alpha=0.7)
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.set_xticks(range(0, 24))

# Save and Show
plt.tight_layout()
plt.savefig('q4_time_analysis_dark.png')
plt.show()