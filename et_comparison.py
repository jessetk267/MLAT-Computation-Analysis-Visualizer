import numpy as np
import matplotlib.pyplot as plt
import math

# Simulation Range
time_step = 0.1
time_start = 0
time_range = 10

# Lorentz Force
e_field = np.array([0, 0, 0])
b_field = np.array([0, 0, 1])
charge = 1
mass = 1

def find_acceleration(q, e, b, m, v):
    # acceleration = (q * (e + np.cross(v, b))) / m
    acceleration = v
    return acceleration

def forward_euler(q, e, b, m, t0, t_end, dt):
    velocity = np.array([1, 0, 0])
    t = t0

    t_values = []
    y_values = []

    while t < t_end: #while loop used since for loop cannot have a float as the step
        t = t + dt

        acceleration = find_acceleration(q, e, b, m, velocity)
        velocity = velocity + dt * acceleration

        magnitude_of_v = np.linalg.norm(velocity)

        t_values.append(t)
        y_values.append(magnitude_of_v)

    return t_values, y_values

def rk4(q, e, b, m, t0, t_end, dt):
    velocity = np.array([1, 0, 0])
    t = t0

    t_values = []
    y_values = []

    while t < t_end: #while loop used since for loop cannot have a float as the step
        t = t + dt
        k1 = dt * find_acceleration(q, e, b, m, velocity)
        k2 = dt * find_acceleration(q, e, b, m, velocity + k1/2)
        k3 = dt * find_acceleration(q, e, b, m, velocity + k2/2)
        k4 = dt * find_acceleration(q, e, b, m, velocity + k3)

        velocity = velocity + (1/6) * (k1 + 2*k2 + 2*k3 + k4)

        magnitude_of_v = np.linalg.norm(velocity)

        t_values.append(t)
        y_values.append(magnitude_of_v)

    return t_values, y_values

def analytical_solution(t0, t_end, dt):
    t = t0

    t_values = []
    y_values = []

    while t < t_end:
        t = t + dt
        magnitude_of_v = math.e ** t

        t_values.append(t)
        y_values.append(magnitude_of_v)

    return t_values, y_values

rk4_t, rk4_y = rk4(charge, e_field, b_field, mass, time_start, time_range, time_step)
fe_t, fe_y = forward_euler(charge, e_field, b_field, mass, time_start, time_range, time_step)
comp_t, comp_y = analytical_solution(time_start, time_range, time_step)

plt.plot(rk4_t, rk4_y, marker='x', label='4th Order Runge-Kutta', linestyle='-')
plt.plot(fe_t, fe_y, marker='o', label='Forward Euler', linestyle='-')
plt.plot(comp_t, comp_y, label='Exact Solution', linestyle='-')

plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Comparison of Euler and RK4 Methods')
plt.legend()
plt.grid()
plt.show()