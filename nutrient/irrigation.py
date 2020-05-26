import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import cross_val_score
from sklearn.metrics import  mean_squared_error
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
#from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
#from sklearn.ensemble import RandomForestRegressor

data=pd.read_csv("test1.csv")
#data

#data.describe()

train_set,test_set=train_test_split(data, test_size=0.2, random_state=42)
#print(len(train_set))

train_set

#data

corr_matrix=data.corr()
corr_matrix['Mean'].sort_values(ascending=False)

attributes=["Result","Mean","Znic","Calcium"]

scatter_matrix(data[attributes],figsize=(12,8))

data.plot(kind="scatter",x="Mean",y="Result",alpha=0.8)

data=train_set.drop(["Result","Phosphorous","Potassium","Nitogen",
                    "Calcium","Sulphur","Znic","Iron","Copper","Cobaalt",
                    "Manganese","Mean"],axis=1)
data_label=train_set[["Result","Phosphorous","Potassium","Nitogen",
                    "Calcium","Sulphur","Znic","Iron","Copper","Cobaalt",
                    "Manganese","Mean"]].copy()

data

#model=LogisticRegression()
#model=LinearRegression()
model=DecisionTreeRegressor()
#model=RandomForestRegressor()

model.fit(data,data_label)

d1=data.iloc[:5]
d1

d2=data_label.iloc[:5]
d2

model.predict(data)

list(d2)

prediction=model.predict(data)
lin_mse=mean_squared_error(data_label,prediction)
lin_rmse=np.sqrt(lin_mse)

lin_mse

score=cross_val_score(model,data,data_label,scoring="neg_mean_squared_error",cv=2)
rms=np.sqrt(-score)

rms

def score(score):
    print("Score :", score)
    print("Mean :",score.mean())
    print("Standard deviation :",score.std())

score(rms)

x_test=test_set.drop(["Result","Phosphorous","Potassium","Nitogen",
                    "Calcium","Sulphur","Znic","Iron","Copper","Cobaalt",
                    "Manganese","Mean"],axis=1)
y_test=test_set[["Result","Phosphorous","Potassium","Nitogen",
                    "Calcium","Sulphur","Znic","Iron","Copper","Cobaalt",
                    "Manganese","Mean"]].copy()
final_result=model.predict(x_test)
final_mse=mean_squared_error(y_test,final_result)
frms=np.sqrt(final_mse)

frms

input=np.array([[4]])
array=model.predict(input)

array[0]

pickle.dump(model,open('finalModel.pkl','wb'))
finalModel=pickle.load(open('finalModel.pkl','rb'))

