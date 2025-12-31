#----------------------
#ΒΙΒΛΙΟΘΗΚΕΣ
#----------------------
import pandas as pd
import numpy as np
import string
from tqdm import tqdm
import spacy
from spacy.lang.el.stop_words import STOP_WORDS
#----------------------
#BHMA 1: ΔΗΜΙΟΥΡΓΙΑ FILE
#----------------------
file_path = r"C:\Users\Alexa\OneDrive\Desktop\mati.csv"

# Διαβάζουμε το αρχείο χωρίς header και ορίζουμε εμείς τις στήλες διότι δεν έχουμε επικεφαλίδες
df = pd.read_csv(
    file_path,
    header=None,
    names=["author_id", "created_at", "geo", "tweet_id", "lang",
        "like_count", "quote_count", "reply_count", "retweet_count",
        "source", "text"])

print(df.head())
print(df.info())

nlp = spacy.load("el_core_news_sm")


#----------------------
# ΒΗΜΑ 2: ΚΑΘΑΡΙΣΜΟΣ ΚΕΙΜΕΝΟΥ
#----------------------
def preprocess_text(text):
    def remove_punctuation_from_words(line):
        words = line.split()
        cleaned_words = [word.translate(str.maketrans("", "", string.punctuation)) for word in words]
        return " ".join(cleaned_words)

    text = text.astype(str).str.lower()
    text = text.str.replace(r"@\w+", "", regex=True)
    text = text.str.replace(r"\brt\b", "", regex=True)
    text = text.str.replace(r"http\S+|www\S+", "", regex=True)
    text = text.str.replace(r"#", "", regex=True)
    text = text.apply(remove_punctuation_from_words)
    text = text.str.replace(r"…", "", regex=True)
    text = text.str.replace(r"[.!?;&]{1,}", "", regex=True)
    text = text.str.replace(r"[.,:\-]", "", regex=True)
    text = text.str.replace(
        r"[\U0001F600-\U0001F64F"  # Emoticons
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
    )
    text = text.str.replace(r"\b\w{1,3}\b", "", regex=True)
    text = text.str.replace(r"\s+", " ", regex=True).str.strip()
    return text

df["clean_text"] = preprocess_text(df["text"])
print("Ο βασικός καθαρισμός του κειμένου ολοκληρώθηκε.")
#----------------------
# ΒΗΜΑ 3:STOPWORDS+LEMMATIZATION
#----------------------
extra_stopwords = {"ειναι", "ηταν", "θα", "να", "και", "με", "για", "στο", "στη", "του", "των", "της"}
custom_stopwords = STOP_WORDS.union(extra_stopwords)

def lemmatize_texts(texts):
    cleaned_texts = []
    for doc in tqdm(nlp.pipe(texts, batch_size=1000, disable=["parser", "ner","tagger"]), total=len(texts), desc="Lemmatization"):
        tokens = [
            token.text for token in doc
            if not token.is_stop and token.is_alpha and len(token.text) >= 4
        ]
        cleaned_texts.append(" ".join(tokens))
    return cleaned_texts

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#df["lemmatized_text"] = lemmatize_texts(df["clean_text"])
df = pd.read_csv("data/mati_preprocessed.csv")
print("Lemmatization is complete.")

#Αποθήκευση preprocessing
import os
os.makedirs("data", exist_ok=True)
df.to_csv("data/mati_preprocessed.csv", index=False)
#-------------------
#ΒΗΜΑ 4:ΕΛΕΓΧΟΣ
#--------------------
print(df.head(3))
print("Τελικό μέγεθος:", df.shape)

#-------------------------------------
#question 1
#--------------------------------------
mati_keywords = [
    "ματι", "πυρκαγια", "φωτια", "νεκρος", "103",
    "ραφηνα", "νεο βουτζα", "κινέτα",
    "εκκενωση", "καμμενος", "τραγωδια",
    "καταστροφη", "θυμα","πνιγμος","λιμενικο"
]

def is_relevant_tweet(text):
    return any(keyword in text for keyword in mati_keywords) #Επιστρέφει κάθε tweet που έχει τις λέξεις κλειδιά

df["lemmatized_text"] = df["lemmatized_text"].fillna("") #Αντικαθιστώ τις κενές τιμές στο Λημματοποιημένο κείμενο
df["is_relevant"] = df["lemmatized_text"].apply(is_relevant_tweet) # Δημιουργούμε καινούργια στήλη


relevant_count = df["is_relevant"].sum() # Πόσα tweets είναι σχετικά
irrelevant_count = len(df) - relevant_count # Πόσα tweets ΔΕΝ είναι σχετικά

print(f"Σχετικά tweets: {relevant_count}")
print(f"ΜΗ σχετικά tweets: {irrelevant_count}")
print(f"Ποσοστό σχετικών: {relevant_count / len(df) * 100:.2f}%")

df[df["is_relevant"]].sample(10)[["text"]]
df[~df["is_relevant"]].sample(10)[["text"]] # Επιλέγω 10 τυχαία δείγματα

df_relevant = df[df["is_relevant"]].copy() # Έφτιαξα DF copy για χρήση σε επόμενες ερωτήσεις
print(" Dataset μετά το filtering:", df_relevant.shape) #Εμφάνιση μεγέθους
df_relevant.to_csv("data/mati_q1_relevant.csv", index=False) #Αποθηκεύει σχετικά tweets σε csv αρχειο
print(df[df["is_relevant"]].sample(10)[["text"]])
