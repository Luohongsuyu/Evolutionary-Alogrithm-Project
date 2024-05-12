import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import time
import os


timerc = []

class Mass:
    def __init__(self, mass, initial_position, initial_velocity):
        self.mass = mass
        self.position = [np.array(initial_position)]
        self.velocity = initial_velocity
        self.net_force = np.array([0.0, 0.0, 0.0])

    def apply_force(self, force):
        self.net_force += force

    def update(self, dt, k_floor=100000.0):  # default floor elastic constant
        acceleration = self.net_force / self.mass
        self.velocity += acceleration * dt
        self.velocity *= 0.999  # Apply the dampening
        new_position = self.position[-1] + self.velocity * dt
        
        # Check collision with the floor
        if new_position[2] <= 0:
            compression = 0 - new_position[2]
            recovery_force = k_floor * compression
            upward_force = np.array([0.0, 0.0, recovery_force])
            acceleration_from_floor = upward_force / self.mass
            self.velocity += acceleration_from_floor * dt
        
        self.position.append(new_position)
        self.net_force = np.array([0.0, 0.0, 0.0])


class Spring:
    def __init__(self, k, mass1, mass2, initial_length, b=0.2, w=2*np.pi, c=0):
        self.k = k
        self.mass1 = mass1
        self.mass2 = mass2
        self.initial_length = initial_length
        self.b = b
        self.w = w
        self.c = c

    def compute_current_length(self, t):
        return self.initial_length + self.b * np.sin(self.w * t + self.c)

    def apply_spring_forces(self, t):
        current_length = self.compute_current_length(t)
        displacement_vector = self.mass2.position[-1] - self.mass1.position[-1]
        displacement_magnitude = np.linalg.norm(displacement_vector)
        force_magnitude = self.k * (displacement_magnitude - current_length)
        force_vector = force_magnitude * (displacement_vector / displacement_magnitude)
        self.mass1.apply_force(force_vector)
        self.mass2.apply_force(-force_vector)

class MassSpringSystem3D:

    def __init__(self, masses, springs, gravity, floor_height, restitution_coefficient, time_step, num_steps):
        self.masses = masses
        self.springs = springs
        self.gravity = gravity
        self.floor_height = floor_height
        self.restitution_coefficient = restitution_coefficient
        self.time_step = time_step
        self.num_steps = num_steps

    def simulate(self):
        for i in range(self.num_steps):
            start_time = time.perf_counter()
            current_time = i * self.time_step
            for spring in self.springs:
                spring.apply_spring_forces(current_time)

            for mass in self.masses:
                gravity_force = mass.mass * self.gravity
                mass.apply_force(gravity_force)
                mass.update(self.time_step, k_floor=100000.0)  # Add the k_floor value here

            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            timerc.append(elapsed_time)

    def plot_animation(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        points = []
        lines = []

        # initialize masses
        for _ in self.masses:
            point, = ax.plot([], [], [], 'o')
            points.append(point)

        # initialize springs
        for _ in self.springs:
            line, = ax.plot([], [], [])
            lines.append(line)

        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_zlim(0, 5)

        def init():
            for point in points:
                point.set_data([], [])
                point.set_3d_properties([])

            for line in lines:
                line.set_data([], [])
                line.set_3d_properties([])

            return points + lines

        def animate(i):
            # update mass
            for j, mass in enumerate(self.masses):
                x = mass.position[i][0]
                y = mass.position[i][1]
                z = mass.position[i][2]
                points[j].set_data(x, y)
                points[j].set_3d_properties(z)

            # update spring
            for j, spring in enumerate(self.springs):
                x_data = [spring.mass1.position[i][0], spring.mass2.position[i][0]]
                y_data = [spring.mass1.position[i][1], spring.mass2.position[i][1]]
                z_data = [spring.mass1.position[i][2], spring.mass2.position[i][2]]
                lines[j].set_data(x_data, y_data)
                lines[j].set_3d_properties(z_data)

            return points + lines

        ani = FuncAnimation(fig, animate, frames=self.num_steps, init_func=init, blit=True, repeat=False, interval=1)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.title("3D Mass-Spring System with Floor and Bouncing Simulation")
        plt.show()




mass1 = Mass(mass=1.0, initial_position=[0.0, 0.0, 0.0], initial_velocity=[0.0, 0.0, 0.0])
mass2 = Mass(mass=1.0, initial_position=[1.0, 0.0, 0.0], initial_velocity=[0.0, 0.0, 0.0])
mass3 = Mass(mass=1.0, initial_position=[0.0, 1.0, 0.0], initial_velocity=[0.0, 0.0, 0.0])
mass4 = Mass(mass=1.0, initial_position=[1.0, 1.0, 0.0], initial_velocity=[0.0, 0.0, 0.0])
mass5 = Mass(mass=1.0, initial_position=[0.0, 0.0, 1.0], initial_velocity=[0.0, 0.0, 0.0])
mass6 = Mass(mass=1.0, initial_position=[0.0, 1.0, 1.0], initial_velocity=[0.0, 0.0, 0.0])
mass7 = Mass(mass=1.0, initial_position=[1.0, 1.0, 1.0], initial_velocity=[0.0, 0.0, 0.0])
mass8 = Mass(mass=1.0, initial_position=[1.0, 0.0, 1.0], initial_velocity=[0.0, 0.0, 0.0])

spring1 = Spring(k=10000.0, mass1=mass1, mass2=mass2, initial_length=1.0)
spring2 = Spring(k=10000.0, mass1=mass1, mass2=mass3, initial_length=1.0)
spring3 = Spring(k=10000.0, mass1=mass2, mass2=mass4, initial_length=1.0)
spring4 = Spring(k=10000.0, mass1=mass3, mass2=mass4, initial_length=1.0)
spring5 = Spring(k=10000.0, mass1=mass1, mass2=mass5, initial_length=1.0)
spring6 = Spring(k=10000.0, mass1=mass2, mass2=mass8, initial_length=1.0)
spring7 = Spring(k=10000.0, mass1=mass3, mass2=mass6, initial_length=1.0)
spring8 = Spring(k=10000.0, mass1=mass4, mass2=mass7, initial_length=1.0)
spring9 = Spring(k=10000.0, mass1=mass5, mass2=mass6, initial_length=1.0)
spring10 = Spring(k=10000.0, mass1=mass5, mass2=mass8, initial_length=1.0)
spring11 = Spring(k=10000.0, mass1=mass6, mass2=mass7, initial_length=1.0)
spring12 = Spring(k=10000.0, mass1=mass7, mass2=mass8, initial_length=1.0)

masses = [mass1, mass2, mass3, mass4, mass5, mass6, mass7, mass8]
springs = [spring1, spring2, spring3, spring4, spring5, spring6, spring7, spring8, spring9, spring10, spring11, spring12]

gravity = np.array([0.0, 0.0, -9.81])
floor_height = 0.0
restitution_coefficient = 0.8
time_step = 0.001
num_steps = 5000

system = MassSpringSystem3D(masses, springs, gravity, floor_height, restitution_coefficient, time_step, num_steps)
system.simulate()

system.plot_animation()