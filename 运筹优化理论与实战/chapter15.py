# -*- coding:utf-8 -*-
# @Time       :2024-05-11 18:39
# @Author     :wjy
# @File       :book_python.py
# @Software   :PyCharm
import pandas as pd
import math
import gurobipy as grb
from gurobipy import *


def cal_best_chuteNumber(flow, m, n, CS):
    """
    计算最优格口数量
    :param flow:
    :param m:
    :param n:
    :param CS:
    :return:
    """

    # 计算最优格口数
    flow['最优格口数'] = flow['货量'].apply(lambda x: 1 if m <= x <= n else int(math.ceil(x / n)))
    # 计算每个格口的流量分配
    flow['格口单位流量'] = flow.apply(lambda x: x['货量'] / x['最优格口数'], axis=1)
    # 计算总的格口数
    sum_chute = flow['最优格口数'].sum()
    # 规范flow 格式
    flow['流向名称'] = flow['流向名称'].astype(int)
    if sum_chute == CS:
        return flow
    else:
        print('需要调整格口阈值')


def solve_problem(flow, CS, inTop, inBottom, out_top, out_bottom):
    """
    调用Gurobi求解该模型
    :param flow:
    :param CS:
    :param inTop:
    :param inBottom:
    :param out_top:
    :param out_bottom:
    :return:
    """

    # 初始化模型
    model = Model('chutePlan')
    # 建立决策变量
    c = {}
    for i in range(1, len(flow) + 1):
        for j in range(1, CS + 1):
            c[i, j] = model.addVar(vtype=GRB.BINARY, name=f'c[{i},{j}]')

    # 两侧逻辑区格口数约束
    model.addConstr(quicksum(c[i, j] for i in range(1, len(flow) + 1) for j in inTop) +
                    quicksum(c[i, j] for i in range(1, len(flow) + 1) for j in out_top) == CS / 2,
                    name='top_chute_constraint')
    model.addConstr(quicksum(c[i, j] for i in range(1, len(flow) + 1) for j in inBottom) +
                    quicksum(c[i, j] for i in range(1, len(flow) + 1) for j in out_bottom) == CS / 2,
                    name='bottom_chute_constraint')
    # 流向需求格口数约束
    for index, row in flow.iterrows():
        model.addConstr(quicksum(c[index + 1, j] for j in inTop) +
                        quicksum(c[index + 1, j] for j in out_top) +
                        quicksum(c[index + 1, j] for j in inBottom) +
                        quicksum(c[index + 1, j] for j in out_bottom) == row['最优格口数'],
                        name=f'chute_constraint_{index + 1}')

    # 均匀分布约束, 引入辅助变量
    diff = {}
    n_top = {}
    n_bottom = {}
    for index, row in flow.iterrows():
        n_top[index + 1] = model.addVar(name=f'top_{index + 1}')
        n_bottom[index + 1] = model.addVar(name=f'bottom_{index + 1}')
        model.addConstr(n_top[index + 1] == quicksum(c[index + 1, j] for j in inTop) +
                        quicksum(c[index + 1, j] for j in out_top), name=f'n_top_{index + 1}_def')
        model.addConstr(n_bottom[index + 1] == quicksum(c[index + 1, j] for j in inBottom) + quicksum(
            c[index + 1, j] for j in out_bottom), name=f'n_bottom_{index + 1}_def')
        diff[index + 1] = model.addVar(name=f'diff_{index + 1}')
        model.addConstr(diff[index + 1] >= n_top[index + 1] - n_bottom[index + 1],
                        name=f'diff_abs_constraint_pos_{index + 1}')
        model.addConstr(diff[index + 1] >= n_bottom[index + 1] - n_top[index + 1],
                        name=f'diff_abs_constraint_neg_{index + 1}')
        model.addConstr(diff[index + 1] <= 1, name=f'upper_bound_{index + 1}')

    # 每个格口都至少分有一个流向
    for j in range(1, CS + 1):
        model.addConstr(quicksum(c[i, j] for i in range(1, len(flow) + 1)) == 1)

    # 添加目标函数，引入辅助变量y
    a_flow = model.addVar(name='a_flow')
    a_flow_expr = LinExpr()
    for index, row in flow.iterrows():
        a_flow_expr += row['格口单位流量'] * \
                  (quicksum(c[index + 1, j] for j in inTop) + quicksum(c[index + 1, j] for j in out_top))
    b_flow = model.addVar(name='b_flow')
    b_flow_expr = LinExpr()
    for index, row in flow.iterrows():
        b_flow_expr += row['格口单位流量'] * \
                  (quicksum(c[index + 1, j] for j in inBottom) + quicksum(c[index + 1, j] for j in out_bottom))
    y = model.addVar(name="y")
    model.addConstr(a_flow == a_flow_expr, 'a_flow_def')
    model.addConstr(b_flow == b_flow_expr, 'b_flow_def')
    model.addConstr(y >= a_flow - b_flow, name='y_abs_constraint_pos')
    model.addConstr(y >= b_flow - a_flow, name='y_abs_constraint_neg')
    model.setObjective(y, GRB.MINIMIZE)
    model.optimize()

    # 获取最优解
    best_sol = {}
    for i in range(1, len(flow) + 1):
        for j in range(1, CS + 1):
            if c[i, j].x != 0:
                best_sol[i, j] = c[i, j].x

    # 统计每个流向所使用的格口并输出
    result_dict = {}
    for key, value in best_sol.items():
        if key[0] in result_dict:
            result_dict[key[0]].append(key[1])
        else:
            result_dict[key[0]] = [key[1]]
    print(result_dict)

    # 输出目标函数值
    print(model.ObjVal)
    print(a_flow.x)
    print(b_flow.x)
    return


if __name__ == '__main__':
    # 参数
    CS = 24  # 设备格口数
    m = 20  # 独立格口阈值
    n = 144  # 分裂格口阈值
    inTop = [1, 2, 3, 4, 5, 6]  # A逻辑区内部格口
    inBottom = [7, 8, 9, 10, 11, 12]  # B逻辑区内部格口
    out_top = [13, 14, 15, 16, 17, 18]  # A逻辑区外部格口
    out_bottom = [19, 20, 21, 22, 23, 24]  # B逻辑区外部格口
    # 创建流向数据
    flow = {'流向名称': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            '货量': [132, 252, 306, 169, 117, 194, 176, 33, 254, 58, 46, 101, 64, 14, 107, 191]
            }
    flow = pd.DataFrame(flow)
    flow = cal_best_chuteNumber(flow, m, n, CS)
    solve_problem(flow, CS, inTop, inBottom, out_top, out_bottom)
