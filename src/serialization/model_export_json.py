import pandas as pd
from sklearn.model_selection import train_test_split

SPLITTER_COEFICIENT = 0.2
SPLITTER_MIDDLE = 0.5
"""
In this algorithm i´m using python 3.11 cause it´s 80% faster than previous versions 
"""
# Reading dataset
df = pd.read_excel("Rest_Mex_Sentiment_Analysis_2023_Train.xlsx")
dfTest = pd.read_excel("Rest_Mex_Sentiment_Analysis_2023_Test.xlsx")

train, test = train_test_split(df, test_size=SPLITTER_COEFICIENT)

testDev, testTest = train_test_split(test, test_size=SPLITTER_MIDDLE)
# parsing to json the df as index form
payload = df.to_json(force_ascii=False, orient='index')

payloadTrain = train.to_json(force_ascii=False, orient='index')

payloadDev = testDev.to_json(force_ascii=False, orient='index')

payloadTestTrain = testTest.to_json(force_ascii=False, orient='index')

payloadTest = dfTest.to_json(force_ascii=False, orient='index')

# Exporting the file as .json
with open("All_RestMext2023_Train.json", "w", encoding='utf-8') as file:
    file.write(payload)

# Exporting the file as .json
with open("Train_RestMext2023_Train.json", "w", encoding='utf-8') as file:
    file.write(payloadTrain)

# Exporting the file as .json
with open("Dev_RestMext2023_Train.json", "w", encoding='utf-8') as file:
    file.write(payloadDev)

with open("Test_RestMext2023_Train.json", "w", encoding='utf-8') as file:
    file.write(payloadTestTrain)

with open("Test_Eval_RestMext2023_Train.json", "w", encoding='utf-8') as file:
    file.write(payloadTest)

