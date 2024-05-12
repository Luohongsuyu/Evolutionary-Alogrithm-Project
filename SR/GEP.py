import geppy as gep
from deap import creator, base, tools
import numpy as np
import random
import operator 
import os
import math
import matplotlib.pyplot as plt
import sympy as sp
import warnings

# 忽略 SymPyDeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)

def simplify_expression(expression, input_names):
    """将GEP表达式转换为sympy表达式并简化
    :param expression: GEP个体的字符串表示
    :param input_names: GEP模型中的输入变量名称列表
    """
    # 创建一个符号变量字典，包括所有输入变量
    symbols = {name: sp.symbols(name) for name in input_names}
    
    # 定义自定义函数到SymPy操作的映射，对于减法和除法使用lambda表达式
    locals_dict = {
        'add': lambda a, b: a + b,
        'sub': lambda a, b: a - b,
        'mul': lambda a, b: a * b,
        'div': lambda a, b: a / b if b != 0 else sp.zoo,  # 防止除以0
        'cos': sp.cos,
        'sin': sp.sin,
        **symbols  # 包含前面创建的所有符号变量
    }
    
    # 将GEP表达式转换为sympy表达式，使用locals_dict处理自定义函数名
    sympy_expr = sp.sympify(expression, locals=locals_dict)
    
    # 简化表达式
    simplified_expr = sp.simplify(sympy_expr)
    
    return simplified_expr

# 在每次运行时使用不同的种子值
s = random.randint(1, 1000)
random.seed(s)
np.random.seed(s)

folder_name = "imagee"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

def load_data(file_path):
    """从文本文件中加载数据并转换为 (x, y) 对的列表格式"""
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # 去除空格和括号
            line = line.strip().replace('(', '').replace(')', '')
            x, y = line.split(',')
            data.append((float(x), float(y)))
    return data

# 调用函数读取数据
file_path = 'fish_angel.txt'  # 将此路径替换为您的文件路径
data_file = load_data(file_path)
data = np.array(data_file)
X = data[:, 0]
Y = data[:, 1]

def protected_div(x1, x2):
    if abs(x2) < 1e-6:
        return 1
    return x1 / x2

def protected_exp(x):
    # 限制指数的输入值，以避免太大的输出
    # 这个限制值可以根据具体情况进行调整
    try:
        # 这里设置了一个安全阈值，可以根据需要调整
        if x > 100:  # e^700 是一个非常大的数，接近浮点数的上限
            return 1000000000 # 返回大数
        elif x < -100:  # 防止产生负的大数
            return 0
        else:
            return math.exp(x)
    except OverflowError:
        return float('inf')  # 如果仍然溢出，则返回无穷大

pset = gep.PrimitiveSet('Main', input_names=['x'])
pset.add_function(operator.add, 2)
pset.add_function(operator.sub, 2)
pset.add_function(operator.mul, 2)
pset.add_function(protected_div, 2)
#pset.add_function(protected_exp, 1)
pset.add_function(math.sin, 1)   # 添加 sin 函数
pset.add_function(math.cos, 1)   # 添加 cos 函数
pset.add_ephemeral_terminal(name='enc', gen=lambda: random.uniform(-1, 1)) # each ENC is a random integer within [-10, 10]


from deap import creator, base, tools

creator.create("FitnessMin", base.Fitness, weights=(-1,))  # to minimize the objective (fitness)
creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMin)


h = 4 
n_genes = 2   

toolbox = gep.Toolbox()
toolbox.register('gene_gen', gep.Gene, pset=pset, head_length=h)
toolbox.register('individual', creator.Individual, gene_gen=toolbox.gene_gen, n_genes=n_genes, linker=operator.add)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register('compile', gep.compile_, pset=pset)

def evaluate(individual):
    func = toolbox.compile(individual)
    Yp = np.array(list(map(func, X)))
    return np.mean((Y - Yp)**2),

toolbox.register('evaluate', evaluate)

toolbox.register('select', tools.selTournament, tournsize=4)

toolbox.register('mut_uniform', gep.mutate_uniform, pset=pset, ind_pb=0.05, pb=1)

toolbox.register('cx_1p', gep.crossover_one_point)
toolbox.register('cx_2p', gep.crossover_two_point)

toolbox.register('mut_ephemeral', gep.mutate_uniform_ephemeral, ind_pb='1p') 
toolbox.pbs['mut_ephemeral'] = 1  


stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

n_pop = 400
n_gen = 50000

pop = toolbox.population(n=n_pop)
hof = tools.HallOfFame(3)   

def log_best_individual(log_file_path, generation, individual, fitness):
    """记录最好的个体和适应度到指定的日志文件"""
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"{generation}, {fitness}\n")

# 运行遗传算法并记录改进的适应度
def run_gep_algorithm():
    # 初始化种群
    pop = toolbox.population(n=n_pop)
    hof = tools.HallOfFame(1)  # 只保留一个最好的个体
    
    log_file_path = 'GEP学习曲线.txt'  # 适应度记录文件
    best_fitness = float('inf')  # 初始化最佳适应度为无穷大
    best_fitness1 = float('inf')

    
    # 开始进化
    for gen in range(n_gen):
        # 评估当前种群中所有个体的适应度
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        
        # 更新Hall of Fame
        hof.update(pop)
        current_best = hof[0]
        current_best_fitness = current_best.fitness.values[0]
        
        # 检查是否需要记录日志
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            log_best_individual(log_file_path, gen, current_best, best_fitness)
        
        # 选择下一代个体
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        # 应用交叉和变异
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.9:
                toolbox.cx_1p(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
                
        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mut_uniform(mutant)
                del mutant.fitness.values
        
        print("gen:",gen,"best fitness:",best_fitness)

        # 检查这一代是否有更好的 fitness1
        current_best = min(pop, key=lambda ind: ind.fitness.values[0])
        current_best_fitness1 = current_best.fitness.values[0]

        if current_best_fitness1 < best_fitness1:
            # 更新历史最佳 fitness1
            best_fitness1 = current_best_fitness1

            # 编译个体以获得可执行的函数
            func = toolbox.compile(current_best)

            # 使用数据中的 x 值计算 y 值
            x_values = [x for x, _ in data]
            y_values = [func(x) for x in x_values]

            # 绘制数据点和当前最佳个体生成的曲线
            plt.scatter(*zip(*data), label='Data Points',s=1)  # 数据点
            plt.plot(x_values, y_values, color='red', label='Best Individual')  # 最佳个体生成的曲线
            plt.title(f"Generation {gen}: Best Individual")
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend()
            plt.savefig(os.path.join(folder_name, f'result_{gen}.png'))
            plt.close()

        # 下一代种群
        pop[:] = offspring
        
    # 打印最终的最佳个体
    best_ind = hof[0]
    print(f'Best Individual: {best_ind}, Fitness: {best_ind.fitness.values[0]}')
    input_names = ['x']  # 假设GEP模型中只有一个输入变量x

    simplified_expr = simplify_expression(best_ind, input_names)
    print(f'Simplified Expression of the Best Individual: {simplified_expr}')

# 执行算法
run_gep_algorithm()
