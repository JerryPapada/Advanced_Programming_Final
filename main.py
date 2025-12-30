import pandas as pd

def load_csv():
    file_path = 'mati.csv'
    try:
        df = pd.read_csv(file_path)

    except FileNotFoundError:
        print("The file was not found. Please check the file path.")

    return df
