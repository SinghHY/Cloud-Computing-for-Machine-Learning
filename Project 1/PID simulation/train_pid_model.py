import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score

data = pd.read_csv("pid_dataset.csv")

X = data[["mean_error","error_std","overshoot"]]
y = data[["Kp","Ki","Kd"]]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4)

model = MultiOutputRegressor(RandomForestRegressor())

model.fit(X_train,y_train)

pred = model.predict(X_test)

print("Model R2 Score:",r2_score(y_test,pred))