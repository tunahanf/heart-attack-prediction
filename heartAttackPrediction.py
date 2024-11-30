import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


df = pd.read_csv("heart.csv")
y = df["output"]
x = df.drop("output", axis=1)
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=.66, random_state=60)

forest = RandomForestClassifier()

model = forest.fit(x_train, y_train)
model.score(x_test,y_test) # model score = 0.8076923076923077

user = input("Enter your variables(1,0,130 etc.) : ") # 31,1,2,130,240,0,0,150,0,2,0,0,2
varList = user.split(",")
varInt = list(map(int, varList))

output = model.predict([varInt])

if output == np.array([0]):
    print("You have LOW risk of heart attack risk")
else:
    print("You have HIGH risk of heart attack")
