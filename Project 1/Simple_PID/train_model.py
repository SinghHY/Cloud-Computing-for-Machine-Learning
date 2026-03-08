import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score

data = pd.read_csv("pid_training_data.csv")

X = data[['error','integral_error','derivative_error']]
y = data[['Kp','Ki','Kd']]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = MultiOutputRegressor(RandomForestRegressor())

model.fit(X_train,y_train)

pred = model.predict(X_test)

score = r2_score(y_test,pred)

print("Model Accuracy (R2):",score)

sample = [[2.1, 1.3, 4.2]]

predicted_gains = model.predict(sample)

print("Predicted PID:", predicted_gains)