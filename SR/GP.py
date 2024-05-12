import operator
import numpy as np
from deap import algorithms, base, creator, tools, gp
import math
import random
import sympy
from sympy import sympify

def simplify_expression(expr_string):
    # 将字符串表达式转换为 sympy 表达式
    sympy_expr = sympify(expr_string)
    # 使用 sympy 的化简功能
    simplified_expr = sympy.simplify(sympy_expr)
    return simplified_expr


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
file_path = 'second_order_undamped.txt'  # 将此路径替换为您的文件路径
data = load_data(file_path)

def evalFitness(individual, points, toolbox):
    func = toolbox.compile(expr=individual)
    sqerrors = ((func(x) - y)**2 for x, y in points)
    return math.fsum(sqerrors) / len(points),len(individual)

# 定义一个生成 [0, 10] 范围内随机数的函数
def random_0_10():
    return random.uniform(0, 10)

def main():
    # 创建适应度类和个体类
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    # 定义操作符
    pset = gp.PrimitiveSet("MAIN", 1) # 新的操作符集合，一个传递进来的参数
    pset.addPrimitive(operator.add, 2) # 加法运算，两位运算符
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(math.sin, 1)   # 添加 sin 函数
    pset.addPrimitive(math.cos, 1)   # 添加 cos 函数
    pset.addEphemeralConstant("rand_0_10", random_0_10)

    # 初始化工具箱
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", evalFitness, points=data, toolbox=toolbox)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    # 运行遗传算法
    pop = toolbox.population(n=200)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    # 打开文件记录学习曲线
    with open('GP学习曲线.txt', 'w') as f:

        previous_min_fitness = float('inf')  # 初始最小适应度设为无限大

        # 初始化进化
        for gen in range(1000):  # 这里以1000代为例


            for i in range(len(pop)):
                if len(pop[i]) > 50:
                    pop[i] = toolbox.individual()

            # 选择下一代
            offspring = toolbox.select(pop, len(pop))
            offspring = algorithms.varAnd(offspring, toolbox, 1, 0.2)

            # 评价个体（有些个体可能已经评价过了）
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # 更新种群
            pop[:] = offspring

            # 更新统计信息并记录
            record = stats.compile(pop)
            hof.update(pop)
            best_ind = hof[0]
            # 仅在找到更好的适应度时记录
            if record['min'] < previous_min_fitness:
                previous_min_fitness = record['min']
                f.write(f"{gen},{record['min']}\n")
            
            print(f"Generation: {gen},Min Fitness: {record['min']}")
            
            # 如果需要，这里可以插入额外的日志记录或者停止条件

    # 获取并显示最优个体
    best_ind = hof[0]
    print("Best individual is:", best_ind)
    print("Fitness:", best_ind.fitness.values[0])

    return pop, stats, hof

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
