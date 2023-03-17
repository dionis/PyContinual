import pandas as pd

"""
In this algorithm i´m using python 3.11 cause it´s 80% faster than previous versions 
"""
# Reading dataset
df = pd.read_excel("Track_Train.xlsx")
# Filtering and parsing to json
with open("classified\\hotel.json", "w", encoding='utf-8') as file:
    file.write(df[df["Attraction"] == "Hotel"].to_json(force_ascii=False, orient='index'))
with open("classified\\restaurant.json", "w", encoding='utf-8') as file:
    file.write(df[df["Attraction"] == "Restaurant"].to_json(force_ascii=False, orient='index'))
with open("classified\\attractive.json", "w", encoding='utf-8') as file:
    file.write(df[df["Attraction"] == "Attractive"].to_json(force_ascii=False, orient='index'))
