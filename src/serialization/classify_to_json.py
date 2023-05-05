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

hotelDf = df[df["Type"] == "Hotel"]
restaurantDf = df[df["Type"] == "Restaurant"]
attractiveDf = df[df["Type"] == "Attractive"]

# Filter
with open("classified\\hotel.json", "w", encoding='utf-8') as file:
    file.write(df[df["Type"] == "Hotel"].to_json(force_ascii=False, orient='index'))

    train, test = train_test_split(hotelDf, test_size=SPLITTER_COEFICIENT)
    testTrain, devTrain = train_test_split(attractiveDf, test_size=SPLITTER_MIDDLE)
    with open("classified\\trainHotel.json", "w", encoding='utf-8') as file:
      file.write(train.to_json(force_ascii=False, orient='index'))
    with open("classified\\devHotel.json", "w", encoding='utf-8') as file:
      file.write(devTrain.to_json(force_ascii=False, orient='index'))
    with open("classified\\testHotel.json", "w", encoding='utf-8') as file:
      file.write(testTrain.to_json(force_ascii=False, orient='index'))

# with open("classified\\testHotel.json", "w", encoding='utf-8') as file:
#     file.write(dfTest[dfTest["Type"] == "Hotel"].to_json(force_ascii=False, orient='index'))

with open("classified\\restaurant.json", "w", encoding='utf-8') as file:
    file.write(df[df["Type"] == "Restaurant"].to_json(force_ascii=False, orient='index'))

    train, test = train_test_split(restaurantDf, test_size=SPLITTER_COEFICIENT)
    testTrain, devTrain = train_test_split(attractiveDf, test_size=SPLITTER_MIDDLE)
    with open("classified\\trainRestaurant.json", "w", encoding='utf-8') as file:
      file.write(train.to_json(force_ascii=False, orient='index'))
    with open("classified\\devRestaurant.json", "w", encoding='utf-8') as file:
      file.write(devTrain.to_json(force_ascii=False, orient='index'))
    with open("classified\\testRestaurant.json", "w", encoding='utf-8') as file:
        file.write(testTrain.to_json(force_ascii=False, orient='index'))

# with open("classified\\testRestaurant.json", "w", encoding='utf-8') as file:
#     file.write(dfTest[dfTest["Type"] == "Restaurant"].to_json(force_ascii=False, orient='index'))

with open("classified\\attractive.json", "w", encoding='utf-8') as file:
    file.write(df[df["Type"] == "Attractive"].to_json(force_ascii=False, orient='index'))

    train, test = train_test_split(attractiveDf, test_size=SPLITTER_COEFICIENT)
    testTrain, devTrain = train_test_split(attractiveDf, test_size=SPLITTER_MIDDLE)
    with open("classified\\trainAttractive.json", "w", encoding='utf-8') as file:
      file.write(train.to_json(force_ascii=False, orient='index'))
    with open("classified\\devAttractive.json", "w", encoding='utf-8') as file:
      file.write(devTrain.to_json(force_ascii=False, orient='index'))
    with open("classified\\testAttractive.json", "w", encoding='utf-8') as file:
        file.write(testTrain.to_json(force_ascii=False, orient='index'))

# with open("classified\\testAttractive.json", "w", encoding='utf-8') as file:
#     file.write(dfTest[dfTest["Type"] == "Attractive"].to_json(force_ascii=False, orient='index'))


