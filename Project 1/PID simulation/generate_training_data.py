import numpy as np
import pandas as pd

dt = 0.1
time = np.arange(0,50,dt)

tau = 10
K = 1.5
T_env = 20
setpoint = 50

dataset = []

for experiment in range(1000):

    Kp = np.random.uniform(0.5,5)
    Ki = np.random.uniform(0.01,1)
    Kd = np.random.uniform(0.01,1)

    temperature = 20
    integral = 0
    previous_error = 0

    errors = []

    for t in time:

        error = setpoint - temperature
        errors.append(error)

        integral += error*dt
        derivative = (error-previous_error)/dt

        u = Kp*error + Ki*integral + Kd*derivative

        dT = (-(temperature-T_env) + K*u)/tau
        temperature += dT*dt

        previous_error = error

    overshoot = (max(errors)/setpoint)*100

    dataset.append([
        np.mean(errors),
        np.std(errors),
        overshoot,
        Kp,
        Ki,
        Kd
    ])

df = pd.DataFrame(dataset,columns=[
    "mean_error",
    "error_std",
    "overshoot",
    "Kp",
    "Ki",
    "Kd"
])

df.to_csv("pid_dataset.csv",index=False)

print("Dataset created")