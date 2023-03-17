import pandas as pd

"""
In this algorithm i´m using python 3.11 cause it´s 80% faster than previous versions 
"""
# Reading dataset
df = pd.read_excel("Track_Train.xlsx")

# parsing to json the df as index form
payload = df.to_json(force_ascii=False, orient='index')

# Exporting the file as .json
with open("RestMext2022.json", "w", encoding='utf-8') as file:
    file.write(payload)

