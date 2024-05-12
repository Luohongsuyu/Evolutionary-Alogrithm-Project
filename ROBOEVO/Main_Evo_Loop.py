import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import math
import random
import copy
import matplotlib.pyplot as plt
import os

dot = []
history = []


class Mass:
    def __init__(self, mass, initial_position, initial_velocity, coefficient_of_restitution=0.1, static_friction_coefficient=0.5, kinetic_friction_coefficient=0.01):
        self.coefficient_of_restitution = coefficient_of_restitution
        self.static_friction_coefficient = static_friction_coefficient
        self.kinetic_friction_coefficient = kinetic_friction_coefficient
        self.mass = mass
        self.position = [np.array(initial_position)]
        self.velocity = initial_velocity
        self.net_force = np.array([0.0, 0.0, 0.0])
        self.gravity = np.array([0.0, 0.0, -9.81*500])
        self.ground_contact_point = None
        self.ground_spring_constant = 100000000

    def apply_force(self, force):
        self.net_force += force

    def update(self, dt):
        acceleration = self.net_force / self.mass
        self.velocity += acceleration * dt
        self.velocity *= 0.999 # Apply the dampening
        new_position = self.position[-1] + self.velocity * dt
        
        # Check if we've just collided with the floor
        just_collided = self.position[-1][2] > 0 and new_position[2] < 0

        # Reflect the velocity if we've just hit the floor
        if just_collided:
            # Reverse the z-component of the velocity
            self.velocity[2] = -self.velocity[2] * self.coefficient_of_restitution
            
            # Additionally dampen the x and y components
            self.velocity[1] *= -0.2
            self.velocity[0] *= -0.2

        # Make sure we don't go below the floor
        if new_position[2] < 0:
            new_position[2] = 0

        self.position.append(new_position)
        self.net_force = np.array([0.0, 0.0, 0.0])



    def calculate_friction_force(self, normal_force, static=False):
        if not static:
            # Calculate kinetic friction
            if np.linalg.norm(self.velocity[:2]) > 1:
                friction_direction = -self.velocity[:2] / np.linalg.norm(self.velocity[:2])
                friction_force_magnitude = self.kinetic_friction_coefficient * np.linalg.norm(normal_force)
            else:
                friction_direction = np.array([0.0, 0.0])
                friction_force_magnitude = 0
        else:
            # Calculate static friction, which opposes the direction of impending motion
            # Assume impending motion is opposite to velocity (for simplicity)
            if np.linalg.norm(self.velocity[:2]) < 10000:
                friction_direction = -self.velocity[:2] / np.linalg.norm(self.velocity[:2])
            else:
                friction_direction = np.array([0.0, 0.0])
            friction_force_magnitude = self.static_friction_coefficient * np.linalg.norm(normal_force)
        
        friction_force = friction_force_magnitude * np.append(friction_direction, 0.0)
        return friction_force
        

class Spring:
    def __init__(self, k, mass1, mass2, initial_length, b=0, w=50*np.pi, c=0):
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
        #print(str(force_magnitude) + '   ' + str(displacement_vector) + '   '+ str(displacement_magnitude) )
        force_vector = force_magnitude * (displacement_vector / displacement_magnitude)
        self.mass1.apply_force(force_vector)
        self.mass2.apply_force(-force_vector)





def calc_ratio(x, c=1):

        a = 0.01  # x增大时y的下限值
        b = 1.0  # x为0时y的值
        y = a + (b - a) * math.exp(-c * x)
        return y

class MassSpringSystem3D:

    def __init__(self, masses, springs, gravity, floor_height, time_step, num_steps):
        self.masses = masses
        self.springs = springs
        self.gravity = gravity
        self.floor_height = floor_height
        self.time_step = time_step
        self.num_steps = num_steps
        self.kinetic_energies = []
        self.potential_energies_gravitational = []
        self.potential_energies_elastic = []
        self.total_energies = []



    def springKs(self, cwcenter, ccwcenter, hardcenter, softcenter,extra4,extra5,extra6,extra7,extra8,extra9,extra10):
        centers = [cwcenter, ccwcenter, hardcenter, softcenter,extra4,extra5,extra6,extra7,extra8,extra9,extra10]
        for spring in self.springs:
            centerdist = []
            center = []
            defk = 0
            defb = 0
            defc = 0
            for idx in range(3):
                val = spring.mass1.position[0][idx] + spring.mass2.position[0][idx]
                center.append(val / 2)
            for cen in centers:
                dist = 0
                for idx in range(3):
                    dist += (center[idx] - cen[idx]) ** 2
                centerdist.append(math.sqrt(dist))
            minidx = np.argmin(centerdist)
            dist_to_closest = centerdist[minidx]
            if dist_to_closest == 0:
                dist_to_closest = 0.00001

            if minidx == 9:
                defk = 15000
                defb = -0.5
                defc = math.pi/2
            elif minidx == 8:
                defk = 35000
                defb = 1.5
                defc = 0
            elif minidx == 7:
                defk = 15000
                defb = 0
                defc = 0
            elif minidx == 6:
                defk = 30000
                defb = 0
                defc = 0
            elif minidx == 5:
                defk = 35000
                defb = 0
                defc = 0
            elif minidx == 4:
                defk = 10000
                defb = 0
                defc = 0
            elif minidx == 3:
                defk = 10000
                defb = -1.5
                defc = 0
            elif minidx == 2:
                defk = 20000
                defb = -1
                defc = math.pi
            elif minidx == 1:
                defk = 20000
                defb = 1
                defc = math.pi
            else : 
                defk = 40000
                defb = 0
                defc = 0

            d  = calc_ratio(dist_to_closest)
            # Adjust the k based on distance
            spring.k = defk * d
            # Adjust the breathing amplitude b based on distance
            spring.b = defb * d
            # Adjust the breathing phase c based on distance
            spring.c = defc * d
            
            
            
    def simulate(self):
        for i in range(self.num_steps):
            current_time = i * self.time_step
            for spring in self.springs:
                spring.apply_spring_forces(current_time)

            for mass in self.masses:
                gravity_force = mass.mass * self.gravity
                mass.apply_force(gravity_force)
                mass.update(self.time_step)  # Add the k_floor value here

        
        p = self.print_finalpos()
        #q = self.calculate_max_spring_length()

        return p 

    def calculate_max_spring_length(self):
        max_length = 0  # 初始最大长度为0

        for spring in self.springs:
            # 计算每个弹簧的长度
            length = np.linalg.norm(spring.mass1.position[-1] - spring.mass2.position[-1])
            max_length = max(max_length, length)

        return max_length
 

    def print_finalpos(self):
        # 初始化一个非常大的数来表示最小差值
        min_dist = float('inf')
        # 初始化变量来存储最接近x=0的mass的x坐标
        closest_x = None

        # 遍历所有的mass
        for mass in self.masses:
            # 计算当前mass的x轴坐标与0的差值
            dist = abs(mass.position[-1][0])
            # 如果这个差值更小，那么更新最小差值和对应的x坐标
            if dist < min_dist:
                min_dist = dist
                closest_x = mass.position[-1][0]

        # 返回最接近x=0的mass的x轴坐标
        return closest_x

    def plot_animation(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        points = []
        lines = []

        # 初始化每个Mass的点
        for _ in self.masses:
            point, = ax.plot([], [], [], 'o')
            points.append(point)

        # 初始化每个Spring的线
        for _ in self.springs:
            line, = ax.plot([], [], [])
            lines.append(line)

        ax.set_xlim(-20, 40)
        ax.set_ylim(-20, 40)
        ax.set_zlim(0, 40)

        def init():
            for point in points:
                point.set_data([], [])
                point.set_3d_properties([])

            for line in lines:
                line.set_data([], [])
                line.set_3d_properties([])

            return points + lines

        def animate(i):
            # 更新每个Mass的点的位置
            for j, mass in enumerate(self.masses):
                x = mass.position[i][0]
                y = mass.position[i][1]
                z = mass.position[i][2]
                points[j].set_data(x, y)
                points[j].set_3d_properties(z)

            # 更新每个Spring的线的位置
            for j, spring in enumerate(self.springs):
                x_data = [spring.mass1.position[i][0], spring.mass2.position[i][0]]
                y_data = [spring.mass1.position[i][1], spring.mass2.position[i][1]]
                z_data = [spring.mass1.position[i][2], spring.mass2.position[i][2]]
                lines[j].set_data(x_data, y_data)
                lines[j].set_3d_properties(z_data)

            return points + lines
     
        ani = FuncAnimation(fig, animate, frames=self.num_steps, init_func=init, blit=True, repeat=False, interval=0)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.title("3D Mass-Spring System with Floor and Bouncing Simulation")
        plt.show()

class centermatrix:
    def __init__(self,cwcenter,ccwcenter,hardcenter,softcenter,extra4,extra5,extra6,extra7,extra8,extra9,extra10,age=0):
        self.cwcenter = cwcenter
        self.ccwcenter = ccwcenter
        self.hardcenter = hardcenter
        self.softcenter = softcenter
        self.extra4 = extra4
        self.extra5 = extra5
        self.extra6 = extra6
        self.extra7 = extra7
        self.extra8 = extra8
        self.extra9 = extra9
        self.extra10 = extra10
        self.age = age

def create_caterpillar(center, cube_size, number_of_cubes):
    masses = []
    shared_masses = None

    for i in range(number_of_cubes):
        # Calculate the center of the new cube
        cube_center = np.array(center) + np.array([i * cube_size, 0, 0])
        new_cube, shared_masses = create_cube(cube_center, cube_size, shared_masses)
        masses.extend(new_cube)

    return masses

def create_cube(center, cube_size, shared_masses):
    half_size = cube_size / 2
    new_masses = []

    # Define the positions of the cube corners
    positions = [np.add(center, (x, y, z)) for x in (-half_size, half_size)
                                              for y in (-half_size, half_size)
                                              for z in (-half_size, half_size)]

    if shared_masses is None:
        # If there are no shared masses, create all 8 masses
        new_masses = [Mass(mass=1.0, initial_position=pos, initial_velocity=[0.0, 0.0, 0.0]) for pos in positions]
    else:
        # Reuse the shared masses and create only 4 new ones
        new_masses.extend(shared_masses)
        new_masses.extend([Mass(mass=1.0, initial_position=pos, initial_velocity=[0.0, 0.0, 0.0]) 
                          for pos in positions[4:]])

    # The last four masses will be shared with the next cube
    shared_masses = new_masses[-4:]
    
    return new_masses, shared_masses

def connect_masses_with_springs(masses, k):
    springs = []
    unique_pairs = set()
    for m1 in masses:
        for m2 in masses:
            # Create a unique identifier for the pair to avoid duplicate springs
            pair_id = tuple(sorted((id(m1), id(m2))))
            if m1 is not m2 and pair_id not in unique_pairs:
                unique_pairs.add(pair_id)
                initial_length = np.linalg.norm(np.array(m1.position) - np.array(m2.position))
                springs.append(Spring(k=k, mass1=m1, mass2=m2, initial_length=initial_length))
    return springs
        


cube_size = 10.0  # Adjust as necessary
k_spring = 1000.0
number_of_cubes = 3

# Caterpillar's masses and springs
gravity = np.array([0.0, 0.0, -9.81*500])
floor_height = 0.0

time_step = 0.0001
num_steps = 2500

"""

caterpillar_masses = []
caterpillar_springs = []

# Start the first cube at half its size above the ground
initial_center = [cube_size / 2, cube_size / 2, cube_size / 2]  

shared_masses = None
for i in range(number_of_cubes):
    cube_center = np.array(initial_center) + np.array([i * cube_size, 0, 0])
    cube_masses, shared_masses = create_cube(cube_center, cube_size, shared_masses)
    caterpillar_masses.extend(cube_masses)
    # Only connect springs within the current cube
    cube_springs = connect_masses_with_springs(cube_masses, k=k_spring)
    caterpillar_springs.extend(cube_springs)

cwcenter = [20.78734744834716, 1.401886842403065, 8.568493252191171]
ccwcenter = [21.15630185596408, 6.850909925027711, 6.999138385715402]
hardcenter = [5.447547155519709, 6.36845756054852, 9.889321473962063]
softcenter = [27.16676736094385, 7.656438902014804, 2.3751803234046287]
extra4 = [20.5999481307084, 3.7493357347042036, 9.8036272215729]
extra5 = [20.307744917188412, 10.590317922433861, 4.826993624553116]
extra6 = [28.251857099135687, 4.490167900681984, 7.6046564112154815]
extra7 = [28.251857099135687, 4.490167900681984, 7.6046564112154815]
extra8 = [28.833987044360477, 1.5714153191146591, 6.3530879514165095]
extra9 = [29.729106871138757, 4.906434748417496, 6.4522358513388784]
extra10 = [16.02996612996516, 1.2497133358285928, 6.3973287699177]
a = centermatrix(cwcenter,ccwcenter,hardcenter,softcenter,extra4,extra5,extra6,extra7,extra8,extra9,extra10)
system = MassSpringSystem3D(caterpillar_masses, caterpillar_springs, gravity, floor_height, time_step, num_steps)
system.springKs(a.cwcenter,a.ccwcenter,a.hardcenter,a.softcenter,a.extra4,a.extra5,a.extra6,a.extra7,a.extra8,a.extra9,a.extra10)
p = system.simulate()
system.plot_animation()
print(p)


"""







def crossover(parent1, parent2):
    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)

    # 随机选择两个切片点
    slice_point1 = random.randint(1, 9)
    slice_point2 = random.randint(slice_point1 + 1, 10)

    # 将父代的属性分成三部分，并交换中间部分
    attributes = ['cwcenter', 'ccwcenter', 'hardcenter', 'softcenter', 'extra4', 'extra5', 'extra6', 'extra7', 'extra8', 'extra9', 'extra10']
    
    for i in range(slice_point1, slice_point2):
        setattr(child1, attributes[i], getattr(parent2, attributes[i]))
        setattr(child2, attributes[i], getattr(parent1, attributes[i]))

    return child1, child2


def fitness(individual, generation):
    caterpillar_masses = []
    caterpillar_springs = []

    # Start the first cube at half its size above the ground
    initial_center = [cube_size / 2, cube_size / 2, cube_size / 2] 

    shared_masses = None
    for i in range(number_of_cubes):
        cube_center = np.array(initial_center) + np.array([i * cube_size, 0, 0])
        cube_masses, shared_masses = create_cube(cube_center, cube_size, shared_masses)
        caterpillar_masses.extend(cube_masses)
        # Only connect springs within the current cube
        cube_springs = connect_masses_with_springs(cube_masses, k=k_spring)
        caterpillar_springs.extend(cube_springs)

    system = MassSpringSystem3D(caterpillar_masses, caterpillar_springs, gravity, floor_height, time_step, num_steps)
    system.springKs(individual.cwcenter,individual.ccwcenter,individual.hardcenter,individual.softcenter,individual.extra4,individual.extra5,individual.extra6,individual.extra7,individual.extra8,individual.extra9,individual.extra10)
    r = system.simulate()
    dot.append((generation,r))
    #system.plot_animation()
    print(r)
    return r


def age_layered_selection(population, max_age_per_layer):
    # 按照年龄分层
    layers = [[] for _ in range(max_age_per_layer)]
    for individual in population:
        age_layer = min(individual.age // max_age_per_layer, max_age_per_layer - 1)
        layers[age_layer].append(individual)

    # 从每层选择个体
    selected_individuals = []
    for layer in layers:
        selected_individuals.extend(random.sample(layer, min(len(layer), len(population) // max_age_per_layer)))

    # 更新年龄
    for individual in population:
        individual.age += 1

    return selected_individuals




def initialize_population(population_size):
    population = []
    for _ in range(population_size):
        individual = centermatrix(
            cwcenter= [random.uniform(0.0, 30.0),random.uniform(0.0, 10.00),random.uniform(0.0, 10.00)],
            ccwcenter= [random.uniform(0.0, 30.0),random.uniform(0.0, 10.00),random.uniform(0.0, 10.00)],
            hardcenter= [random.uniform(0.0, 30.0),random.uniform(0.0, 10.00),random.uniform(0.0, 10.00)],
            softcenter= [random.uniform(0.0, 30.0),random.uniform(0.0, 10.00),random.uniform(0.0, 10.00)],
            extra4= [random.uniform(0.0, 30.0),random.uniform(0.0, 10.00),random.uniform(0.0, 10.00)],
            extra5= [random.uniform(0.0, 30.0),random.uniform(0.0, 10.00),random.uniform(0.0, 10.00)],
            extra6= [random.uniform(0.0, 30.0),random.uniform(0.0, 10.00),random.uniform(0.0, 10.00)],
            extra7= [random.uniform(0.0, 30.0),random.uniform(0.0, 10.00),random.uniform(0.0, 10.00)],
            extra8= [random.uniform(0.0, 30.0),random.uniform(0.0, 10.00),random.uniform(0.0, 10.00)],
            extra9= [random.uniform(0.0, 30.0),random.uniform(0.0, 10.00),random.uniform(0.0, 10.00)],
            extra10= [random.uniform(0.0, 30.0),random.uniform(0.0, 10.00),random.uniform(0.0, 10.00)],

        )
        population.append(individual)
    return population

def mutate(individual, mutation_rate):
    if random.random() < mutation_rate:
        k = random.randint(0,10)
        if k == 0:
            for i in range(0,3):
                individual.cwcenter[i] += random.uniform(0,1)
        if k == 1:
            for i in range(0,3):
                individual.ccwcenter[i] += random.uniform(0,1)
        if k == 2:
            for i in range(0,3):
                individual.hardcenter[i] += random.uniform(0,1)
        if k == 3:
            for i in range(0,3):
                individual.softcenter[i] += random.uniform(0,1)
        if k == 4:
            for i in range(0,3):
                individual.extra4[i] += random.uniform(0,1)
        if k == 5:
            for i in range(0,3):
                individual.extra5[i] += random.uniform(0,1)
        if k == 6:
            for i in range(0,3):
                individual.extra6[i] += random.uniform(0,1)
        if k == 7:
            for i in range(0,3):
                individual.extra7[i] += random.uniform(0,1)
        if k == 8:
            for i in range(0,3):
                individual.extra8[i] += random.uniform(0,1)
        if k == 9:
            for i in range(0,3):
                individual.extra9[i] += random.uniform(0,1)
        if k == 10:
            for i in range(0,3):
                individual.extra10[i] += random.uniform(0,1)
        
    return individual

def run_evolution(population_size, mutation_rate, generations, max_age_per_layer=5, new_individuals_rate=0.1):
    # Initialize population
    population = initialize_population(population_size)
    best_fitness = -100000
    for generation in range(generations):
        # Evaluate fitness for each individual
        fitness_results = [(individual, fitness(individual,generation)) for individual in population]

        # Age Layered Selection
        parents = age_layered_selection(population, max_age_per_layer)

        # Crossover and Mutation
        children = []
        while len(children) < population_size:
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = crossover(parent1, parent2)
            children.extend([child1, child2])

        # Mutate
        mutated_children = [mutate(child, mutation_rate) for child in children]

        # Add new individuals
        num_new_individuals = int(population_size * new_individuals_rate)
        new_individuals = initialize_population(num_new_individuals)

        # Update population
        population = mutated_children[:population_size - num_new_individuals] + new_individuals

        # Find the best individual in this generation
        best_individual_generation = max(fitness_results, key=lambda x: x[1])[0]
        best_fitness_generation = max(fitness_results, key=lambda x: x[1])[1]

        for individual, fit in fitness_results:
            if fit > best_fitness:
                best_fitness = fit
                history.append((generation,fit))

        print(f"Generation {generation}: Best Fitness = {best_fitness_generation}")

    # After all generations, return the best individual of the last generation
    return best_individual_generation





best_individual = run_evolution(population_size=30, mutation_rate=0.1, generations=50)





# 使用当前文件夹路径，设置要输出的文件名
file_name1 = 'dot.txt'
file_name2 = 'learncurve.txt'

    # 打开文件，并将数组中的内容写入文件
with open(file_name1, 'w') as file:
    for item in dot:
        file.write(str(item) + '\n')

with open(file_name2, 'w') as file:
    for item in history:
        file.write(str(item) + '\n')




print(best_individual.cwcenter)
print(best_individual.ccwcenter)
print(best_individual.hardcenter)
print(best_individual.softcenter)
print(best_individual.extra4)
print(best_individual.extra5)
print(best_individual.extra6)
print(best_individual.extra7)
print(best_individual.extra8)
print(best_individual.extra9)
print(best_individual.extra10)