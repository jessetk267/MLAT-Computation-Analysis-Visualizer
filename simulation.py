import numpy as np
import matplotlib.pyplot as plt

# Constants
me = 9.10938356E-31 # Mass of electron (kg)
q = -1.602176634E-19 # Charge of electron (C)


def analytic_larmor_gyration_ic(t_values, r_L, omega, q_sign):
    """
    Analytic gyration for uniform Bz, E=0 with initial:
    r(0) = (0,0), v(0) = (0, v0)
    q_sign = np.sign(q) (electron: -1)
    """
    theta = omega * t_values
    x = q_sign * r_L * (1.0 - np.cos(theta))
    y = r_L * np.sin(theta)
    return x, y

# Field definitions
def get_B(x, y, z):
    return np.array([0.0, 0.0, 0.1]) # 0.1 Tesla in z-direction

def get_E(x, y, z):
    return np.array([0.0, 0.0, 0.0]) # Zero E-field

def get_acc(pos, vel):
    E = get_E(*pos)
    B = get_B(*pos)
    # Lorentz force: F = q(E + v x B)
    return (q / me) * (E + np.cross(vel, B))

def RK4_func(r, v, dt):
    return ( v * dt ), ( dt * get_acc(r, v) )

def Euler_push(x, y, z, vx, vy, vz, dt):
    r = np.array([x, y, z])
    v = np.array([vx, vy, vz])
    E = get_E(*r)
    B = get_B(*r)
    # Forward Euler uses current time velocity to compute acceleration
    a = get_acc(r, v)

    v_new = v + a * dt
    r_new = r + v * dt

    return r_new[0], r_new[1], r_new[2], v_new[0], v_new[1], v_new[2]

def RK4_push(x, y, z, vx, vy, vz, dt):
    r = np.array([x, y, z])
    v = np.array([vx, vy, vz])

    k1_p, k1_v = RK4_func(r, v, dt)
    k2_p, k2_v = RK4_func(r + k1_p/2, v + k1_v/2, dt)
    k3_p, k3_v = RK4_func(r + k2_p / 2, v + k2_v / 2, dt)
    k4_p, k4_v = RK4_func(r + k3_p, v + k3_v, dt)

    v_new = v + (1 / 6) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

    r_new = r + (1 / 6) * (k1_p + 2 * k2_p + 2 * k3_p + k4_p)

    return r_new[0], r_new[1], r_new[2], v_new[0], v_new[1], v_new[2]

def Boris_push(x, y, z, vx, vy, vz, dt, frequency_correction=True):
    r = np.array([x, y, z])
    v = np.array([vx, vy, vz])
    e_field = get_E(x, y, z)
    b_field = get_B(x, y, z)

    qmdt2 = (q * dt) / (2 * me)

    #complete this part.
    v_dash = v + qmdt2 * e_field

    t = qmdt2 * b_field
    s = ( 2 * t ) / (1 + np.dot(t, t))

    v_prime = v_dash + np.cross(v_dash, t)
    v_plus = v_dash +  np.cross(v_prime, s)

    v_new = v_plus + qmdt2 * e_field

    r_new = r + v_new * dt

    return r_new[0], r_new[1], r_new[2], v_new[0], v_new[1], v_new[2]

def main():
    # Initial Conditions
    x0, y0, z0 = 0.0, 0.0, 0.0
    vx0, vy0, vz0 = 0.0, 1.0E6, 0.0

    # Physics Params
    B_mag = 0.1
    Larmor_radius = (me * vy0) / (abs(q) * B_mag)
    Larmor_freq = (abs(q) * B_mag) / me
    Period = 2 * np.pi / Larmor_freq

    print(f"Larmor Radius: {Larmor_radius:.4e} m")
    print(f"Larmor Frequency: {Larmor_freq:.4e} rad/s")
    print(f"Period of gyration: {Period:.4e} s")
    # Simulation Settings
    N_periods = 1000 # Run long enough to see drifts
    t_end = N_periods * Period
    dt = Period / 10 # 20 steps per cycle (Standard resolution)
    t_values = np.arange(0, t_end+dt, dt)

    # Dictionary to store results
    solvers = {
        "Euler": Euler_push,
        # "Leapfrog": Leapfrog_push,
        "RK4": RK4_push,
        "Boris": Boris_push
    }
    results = {}

    for name, func in solvers.items():
        x, y, z = x0, y0, z0
        vx, vy, vz = vx0, vy0, vz0
        x_hist, y_hist = [], []
        energy_hist = [] # Kinetic Energy
        for i, t in enumerate(t_values):
            if name == "Boris" and i == 0:
                _, _, _, vx, vy, vz = Boris_push(x, y, z, vx, vy, vz, -dt/2)
            x_hist.append(x)
            y_hist.append(y)
            v_mag_sq = vx**2 + vy**2 + vz**2
            energy_hist.append(0.5 * me * v_mag_sq)
            x, y, z, vx, vy, vz = func(x, y, z, vx, vy, vz, dt)

        results[name] = {"x": x_hist, "y": y_hist, "E": energy_hist}

    # ==========================================
    # Visualization
    # ==========================================
    # --- Analytical orbit ---
    omega = Larmor_freq
    x_ana, y_ana = analytic_larmor_gyration_ic(
        t_values=t_values,
        r_L=Larmor_radius,
        omega=omega,
        q_sign=np.sign(q)
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: XY Trajectory
    ax1 = axes[0]
    ax1.plot(x_ana, y_ana, 'k', linewidth=2, label='Analytical')

    for name, data in results.items():
        if name == "Euler":
            # Euler explodes too fast, plot only first few points or limit axis
            ax1.plot(data["x"], data["y"], label=name, linestyle='--', alpha=0.5)
        else:
            ax1.plot(data["x"], data["y"], label=name, linestyle='--', alpha=0.5)

    ax1.set_title(f'Electron Trajectory (N={N_periods} periods)')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.axis('equal')
    ax1.grid(True)
    ax1.legend()
    # Set limits to see the stable orbits (Euler might be way off screen)
    limit = Larmor_radius * 1.5
    ax1.set_xlim(-2*limit, limit)
    ax1.set_ylim(-limit, limit)

    # Plot 2: Energy Conservation (Normalized)
    ax2 = axes[1]
    for name, data in results.items():
        E0 = data["E"][0]
        E_norm = np.array(data["E"]) / E0
        ax2.plot(t_values, E_norm, label=name)

    ax2.set_title('Normalized Energy (E/E0) vs Time')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Energy Ratio')
    ax2.grid(True)
    ax2.legend()
    # Zoom in to see RK4 vs Boris stability
    ax2.set_ylim(0.5, 2.0)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()