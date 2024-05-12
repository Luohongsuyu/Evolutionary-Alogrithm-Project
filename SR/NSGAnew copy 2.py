import operator
import numpy as np
from deap import algorithms, base, creator, tools, gp
import math
import random
import sympy
from sympy import sympify
import copy
import matplotlib.pyplot as plt
import os

# 在每次运行时使用不同的种子值
s = random.randint(1, 1000)
random.seed(s)
np.random.seed(s)


folder_name = "image"
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
data = load_data(file_path)


def max_trig_nesting_depth(individual):
    check = str(individual)
    # 定义特定子字符串列表
    substrings = ['sin(cos', 'sin(sin', 'cos(cos', 'cos(sin','sin(protected_exp(','protected_exp(sin(']
    # 初始化计数器
    total_count = 0


    # 遍历每个子字符串，计算出现次数并累加到总计数器
    for substring in substrings:
        count = check.count(substring)
        total_count += count
    
    return 20*total_count

    
# 计算数据点导数的符号
data_deriv_signs = np.sign([data[i+1][1] - data[i][1] for i in range(len(data)-1)])
   

def evalFitness(individual, points, toolbox,data_deriv_signs):
    func = toolbox.compile(expr=individual)
    # 计算平方误差
    sqerrors = ((func(x) - y)**2 for x, y in points)
    fitness1 = math.fsum(sqerrors) / len(points)
    
    
    # 计算估计方程在数据点导数的符号
    estimated_deriv_signs = np.sign([func(points[i+1][0]) - func(points[i][0]) for i in range(len(points)-1)])
    
    # 计算导数符号匹配的适应度（不匹配的数量）
    fitness2 = sum(data_deriv_signs != estimated_deriv_signs)
    
    return fitness1, fitness2

# 定义一个生成 [0, 10] 范围内随机数的函数
def random_0_1():
    a = [0.5,15,math.pi/2,15*math.sqrt(1 - 0.5**2),1,2]
    return random.choice(a)

def random_01():
    
    return random.uniform(0, 1)

def pi():
    return math.pi

def protected_div(x1, x2):
    if abs(x2) < 1e-6:
        return 1000000000
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

def main():


    # 创建适应度类和个体类
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))

    # weights 你优化的目标，和你要优化的权重
    # 多目标优化就代表我不只优化一个点，是多个点，在所有符号回归里面，都有一个点是一定要优化的，叫做均方误差MSE (mean square error)变小
    # 第二个优化的点叫做算式的复杂度，我们希望要复杂度越小越好(在保证MSE不变大的情况下)：
    # 在MSE一样的情况下我们认为x*sin(x)*cos(x)比后面这个（x^2+x+2）*sin（x^2+x+2）好
    # 复杂度这个东西，可能是因人而异的，所以标准怎么取更好呢，我们就要做实验了
    # weights=(-1.0, -1.0) 括号里面，有几个数，代表我们要优化几个目标，负数代表着我优化的这个目标是越小越好，正数代表越大越好。具体数值代表什么，代表这几个目标之间的重要程度关系。比如(-10,-1)代表着后面那个不如前面重要
    # base.Fitness代表这两个优化的目标都是fitness，我们需要在fitness方程里反应这个顺序和具体的东西。
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)

    # 定义操作符
    pset = gp.PrimitiveSet("MAIN", 1) # 新的操作符集合，一个传递进来的参数
    pset.addPrimitive(operator.add, 2) # 加法运算，两位运算符
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protected_div, 2)
    pset.addPrimitive(protected_exp, 1)   # 添加自然指数函数，一个参数
    pset.addPrimitive(math.sin, 1)   # 添加 sin 函数
    pset.addPrimitive(math.cos, 1)
    #pset.addEphemeralConstant("rand_0_1", random_0_1)
    pset.addEphemeralConstant("rand_01", random_01)
    
    


    # 初始化工具箱
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=5)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", evalFitness, points=data, toolbox=toolbox, data_deriv_signs = data_deriv_signs)
    toolbox.register("select", tools.selNSGA2)
    #toolbox.register("select_t", tools.selTournament, tournsize=3)
    #toolbox.register("select", tools.selNSGA3, ref_points=tools.uniform_reference_points(nobj=2, p=12))
    toolbox.register("mate", gp.cxOnePoint) 
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.register("mutate_node", gp.mutNodeReplacement, pset=pset)  # 节点替换变异
    toolbox.register("mutate_mutEphemeral", gp.mutEphemeral, mode = "one")  # 常数变异


    # 初始化用于存储每代适应度信息的列表
    generation_fitness = []
    # 初始化最佳个体适应度
    best_fitness1 = float('inf')

    # 创建初始种群
    pop = toolbox.population(n=400)
    hof = tools.HallOfFame(1)

    # 评估初始种群的个体
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # 这里是我们的遗传算法循环
    for gen in range(100000):  # 假设总共运行1000代
        # 选择下一代个体
        
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # 应用交叉
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

        # 应用变异
        for mutant in offspring:
        
            if len(mutant)>=200:
                
                new_individual = toolbox.individual()
                # 替换原个体
                offspring[offspring.index(mutant)] = new_individual
                # 重新计算适应度值
                del new_individual.fitness.values
                
                

            if random.random() < 0.3:
                toolbox.mutate(mutant)               
                del mutant.fitness.values

        # 评估需要重新计算适应度的个体
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # 替换种群
        pop[:] = offspring

        # 更新 Hall of Fame
        hof.update(pop)

        if gen % 50 ==0:
            # 记录这一代的适应度信息
            generation_fitness.append([ind.fitness.values for ind in pop])
            # 获取并显示最优个体的表达式
            best_ind = hof[0]
            print("Generation:", gen ,"Fitness: ", best_ind.fitness.values[0], best_ind.fitness.values[1])
        
        if gen % 500 == 0:
            with open('点图数据_100000代', 'a') as file:
                for ind in pop:
                    file.write(f"{gen}, {ind.fitness.values}\n")

        # 检查这一代是否有更好的 fitness1
        current_best = min(pop, key=lambda ind: ind.fitness.values[0])
        current_best_fitness1 = current_best.fitness.values[0]

        if current_best_fitness1 < best_fitness1:
            # 更新历史最佳 fitness1
            best_fitness1 = current_best_fitness1

            # 编译个体以获得可执行的函数
            func = toolbox.compile(expr=current_best)

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
            # 记录到文件
            with open('学习曲线数据_100000代', 'a') as file:
                file.write(f"{gen}, {best_fitness1}\n")



    # 找到 Pareto 前沿的个体
    pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]

    # 打开文件以写入结果
    with open("result.txt", "w") as file:
        for ind in pareto_front:
            # 获取个体的表达式
            expr = str(ind)
            # 获取个体的适应度值
            fitness1, fitness2= ind.fitness.values
            # 写入文件
            file.write(f"{expr}, {fitness1}, {fitness2}\n")

    # [其余代码保持不变]
    
    best_ind = hof[0]

    print('best individual: ')
    print(best_ind)
    print("Fitness: ", best_ind.fitness.values[0])

    return pop, generation_fitness, hof

if __name__ == "__main__":
    main()