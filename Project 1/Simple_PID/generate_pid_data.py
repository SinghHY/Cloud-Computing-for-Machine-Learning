import numpy as np
import pandas as pd

np.random.seed(0)

N = 5000

error = np.random.uniform(-10,10,N)
integral_error = np.random.uniform(-50,50,N)
derivative_error = np.random.uniform(-5,5,N)

Kp = 0.6 + 0.08*error + np.random.normal(0,0.05,N)
Ki = 0.04 + 0.01*integral_error + np.random.normal(0,0.01,N)
Kd = 0.12 + 0.05*derivative_error + np.random.normal(0,0.02,N)

data = pd.DataFrame({
    "error": error,
    "integral_error": integral_error,
    "derivative_error": derivative_error,
    "Kp": Kp,
    "Ki": Ki,
    "Kd": Kd
})

data.to_csv("pid_training_data.csv", index=False)

print("Dataset created!")
print(data.head())

