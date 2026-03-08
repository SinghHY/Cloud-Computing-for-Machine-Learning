import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
dt = 0.1
time = np.arange(0,50,dt)

setpoint = 50
temperature = 20

# PID gains
Kp = 2.0
Ki = 0.2
Kd = 0.5

integral = 0
previous_error = 0

tau = 10
K = 1.5
T_env = 20

temp_history = []
control_history = []

for t in time:

    error = setpoint - temperature

    integral += error*dt
    derivative = (error - previous_error)/dt

    u = Kp*error + Ki*integral + Kd*derivative

    # plant model
    dT = (-(temperature-T_env) + K*u)/tau
    temperature += dT*dt

    temp_history.append(temperature)
    control_history.append(u)

    previous_error = error


plt.figure()
plt.plot(time,temp_history,label="Temperature")
plt.axhline(setpoint,linestyle="--",label="Setpoint")
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.title("PID Temperature Control")
plt.legend()
plt.show()


temp_array = np.array(temp_history)

# Overshoot
max_temp = np.max(temp_array)
overshoot = ((max_temp-setpoint)/setpoint)*100

# Settling time
tolerance = 0.02*setpoint
settling_time = None

for i in range(len(temp_array)):
    if np.all(np.abs(temp_array[i:] - setpoint) < tolerance):
        settling_time = time[i]
        break

# steady-state error
steady_state_error = abs(temp_array[-1] - setpoint)

print("Overshoot:",overshoot,"%")
print("Settling Time:",settling_time)
print("Steady State Error:",steady_state_error)