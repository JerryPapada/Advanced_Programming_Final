import pandas as pd
import numpy as np

# ----------------------
# ΦΟΡΤΩΣΗ ΣΧΕΤΙΚΩΝ TWEETS (Q1)
# ----------------------
df = pd.read_csv("data/mati_q1_relevant.csv") # φορτώνω τα tweets που με ενδιαφερουν

# ----------------------
# ΥΠΟΛΟΓΙΣΜΟΣ TWEET VOLUME ΑΝΑ AUTHOR
# ----------------------
# Ομαδοποιούμε τα tweets ανά author
# και υπολογίζουμε πόσα tweets έχει δημοσιεύσει ο καθένας.
author_volume = (
    df.groupby("author_id")
    .size()
    .reset_index(name="total_tweets"))

# ----------------------
# THRESHOLD >= 3 tweets
# ----------------------
# Αφαιρούμε authors με λιγότερα από τρία tweets
active_authors = author_volume[
    author_volume["total_tweets"] >= 3
]["author_id"]

df_active = df[df["author_id"].isin(active_authors)].copy() #είχαμε warning χωρίς το copy


print("Σύνολο authors:", author_volume.shape[0])
print("Ενεργοί authors (>=3 tweets):", df_active["author_id"].nunique()) #NUMBER of unique values(διαφορετικοι authors)
print("Tweets που κρατήθηκαν:", len(df_active))
# ----------------------
# ΥΠΟΕΡΩΤΗΜΑ 1:
# Total tweets per active author
# ----------------------
#Υπολογίζουμε τον συνολικό αριθμό tweets για κάθε ενεργό author και τους ταξινομούμε κατά φθίνουσα σειρά.
author_tweet_counts = (
    df_active.groupby("author_id")
    .size()
    .reset_index(name="total_tweets")
    .sort_values("total_tweets", ascending=False)
)

print("\nTop 20 active authors by total tweets:")
print(author_tweet_counts.head(20))

print("\nBottom active authors (still >=3 tweets):")
print(author_tweet_counts.tail(20))

#---------------------------
# ΥΠΟΕΡΩΤΗΜΑ 2
#---------------------------
#χωρίς datetime:
#δεν μπορώ να αφαιρέσω χρόνους
#δεν υπάρχουν χρονικά metrics
df_active["created_at"] = pd.to_datetime(
    df_active["created_at"],
    utc=True,
    errors="coerce"   #Δημιουργώ NaN τιμές εκεί που δεν μπορεί να γίνει Datetime για να τις dropαρω μετά
)

df_active = df_active.dropna(subset=["created_at"]) #πότε ξεκίνησε και πότε τελείωσε η δραστηριότητα κάθε author
                                                    #subset=κοίτα μόνο τη γραμμή created at
author_time_span = (
    df_active
    .groupby("author_id")["created_at"] #ομαδοποιώ τα tweets κάθε author και κοιτάζω ημερομηνία
    .agg(
        first_tweet="min",
        last_tweet="max"
    )
    .reset_index() #Το ξανακάνω στήλη
)

author_time_span["time_span_days"] = (
    author_time_span["last_tweet"] -
    author_time_span["first_tweet"]
).dt.days #παίρνω απο το timedate μόνο την ημέρα

author_time_span = author_time_span.sort_values(
    "time_span_days",
    ascending=False
)

print("\nAuthors with largest activity time span:")
print(author_time_span.head(20))

print("\nAuthors with smallest activity time span:")
print(author_time_span.tail(20))

#-------------------------------------
# ΥΠΟΕΡΩΤΗΜΑ 3
#------------------------------------

df_active = df_active.sort_values(
    ["author_id", "created_at"]) #σορτάρω ανά author και μετά ανά ώρα
df_active["time_diff"] = (
    df_active
    .groupby("author_id")["created_at"]
    .diff()     #πχ created_at : 14:00 , 14:05 , 14:15 το κάνει NaT , 5λεπτά , 15 λεπτά
)  #βρίσκει:κάθε πόσο χρόνο ποστάρει ο author ( αγνοεί το πρώτο tweet )

avg_time_gap = (
    df_active
    .groupby("author_id")["time_diff"]
    .mean()     #Υπολογίζουμε τον μέσο χρονικό διάστημα μεταξύ διαδοχικών tweets για κάθε author.
    .dt.total_seconds() / 60) # dt για να πάρω αριθμούς. πχ το 00:15:00 γινεται 900 λεπτα και μετά διαιρώ με 60

avg_time_gap = avg_time_gap.reset_index(
    name="avg_time_gap_minutes")   #δίνω όνομα στη στήλη

print("\nAuthors with smallest avg time gap (very frequent posting):")
print(avg_time_gap.sort_values("avg_time_gap_minutes").head(20))

print("\nAuthors with largest avg time gap (sparse posting):")
print(avg_time_gap.sort_values("avg_time_gap_minutes", ascending=False).head(20))

#-------------------------------
# WEIGHTED SCORING SYSTEM
#----------------------------------

#ΒΗΜΑ 1: ΕΝΩΣΗ ΟΛΩΝ ΤΩΝ METRICS
author_stats = (
    author_tweet_counts
    .merge(author_time_span, on="author_id")   #ενώνω τους πίνακες
    .merge(avg_time_gap, on="author_id")
)
#ΒΗΜΑ 2: ΚΑΘΑΡΙΣΜΟΣ ΤΙΜΩΝ
author_stats["avg_time_gap_minutes"] = (
    author_stats["avg_time_gap_minutes"]
    .fillna(author_stats["avg_time_gap_minutes"].median()) #fillna(median) -->«αν δεν ξέρω το gap ενός author,
    .replace(0, 0.1)  #κανω διαίρεση μετά                  # βάλε μια τυπική τιμή»
)
#ΒΗΜΑ 3: ΟΡΙΣΜΟΣ WEIGHTED SCORE
author_stats["activity_score"] = (
    0.5 * np.log1p(author_stats["total_tweets"]) +
    0.3 * np.log1p(author_stats["time_span_days"] + 1) +
    0.2 * (1 / np.log1p(author_stats["avg_time_gap_minutes"]))
)   #log έβαλα για να μην σπάσουν οι τιμές.Δεν μας ενδιαφέρει το spam,μας ενδιαφέρει ο ενεργός χρήστης
#ΒΗΜΑ 4: RANKING AUTHORS
author_stats = author_stats.sort_values(
    "activity_score", ascending=False
)

print("\nTop 20 authors by activity score:")
print(author_stats.head(20))

print("\nLeast active (still >=3 tweets):")
print(author_stats.tail(20))

# ----------------------
# TABLE: Top Active Authors
# ----------------------

top_authors = author_stats[[
    "author_id",
    "total_tweets",
    "time_span_days",
    "avg_time_gap_minutes",
    "activity_score"
]].head(20)

print("\nTop 20 Most Active Authors:")
print(top_authors)

# Αποθήκευση για το report
top_authors.to_csv(
    "data/q3_top20_active_authors.csv",
    index=False
)

# ----------------------
# VISUALIZATION: Scatter Plot
# ----------------------
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

plt.scatter(
    author_stats["total_tweets"],
    author_stats["activity_score"],
    alpha=0.5
)

plt.xlabel("Total Tweets per Author")
plt.ylabel("Activity Score")
plt.title("Author Activity: Tweet Volume vs Activity Score")

plt.tight_layout()
plt.savefig("q3_scatter_volume_vs_activity_score.png")
plt.show()