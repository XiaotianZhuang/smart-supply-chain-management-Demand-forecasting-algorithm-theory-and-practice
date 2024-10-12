from gurobipy import *
import pandas as pd
import os

file_path = os.path.dirname(__file__) + "/分拣功能定位.xlsx"
sort_df = pd.read_excel(file_path, sheet_name='分拣场地信息')
demand_df = pd.read_excel(file_path, sheet_name='需求信息')
path_df = pd.read_excel(file_path, sheet_name='线路信息')

L = {(row['线路起点'], row['线路终点']): {'max_capacity': row['单车最大装载量'], 'trans_cost': row['单车运输成本'],
                                          'path_type': row['线路类型']} for index, row in path_df.iterrows()}
UD = {(row['需求对起点'], row['需求对终点']): {'demand': row['货量需求']} for index, row in demand_df.iterrows()}
S = {row['分拣场地']: {'sort_type': row['场地功能'].split(','), 'opre_cost': row['单包分拣操作成本'],
                       'max_path': [int(x) for x in row['最多线路数'].split(',')], 'max_volume': row['总处理货量上限']}
     for index, row in sort_df.iterrows()}

# 生成候选路由
# 从L构建图的邻接表表示
graph = {}
for (start, end) in L.keys():
    if start not in graph:
        graph[start] = []
    graph[start].append(end)


def dfs_paths(graph, start, end, path=None, max_length=4):
    if path is None:
        path = [start]
    if start == end:
        return [path]
    if start not in graph or len(path) >= max_length:  # 如果路径已达到最大长度，停止搜索
        return []
    paths = []
    for node in graph[start]:
        if node not in path:
            new_paths = dfs_paths(graph, node, end, path + [node], max_length)
            for new_path in new_paths:
                paths.append(new_path)
    return paths


# 初始化一个集合来存储所有线路
all_routes_list = []
# 使用DFS搜索候选路由
for key in UD.keys():
    start, end = key
    all_paths = dfs_paths(graph, start, end)
    # 将路径转换为所需的格式，例如 A-B-C 转换为 [(A, B), (B, C)]
    formatted_paths = [[(path[i], path[i + 1]) for i in range(len(path) - 1)] for path in all_paths]
    # 更新UD字典
    UD[key]['候选路由'] = formatted_paths
    # 将候选路由转换为字符串格式
for key, value in UD.items():
    # 将每个路由的每一段使用"->"连接，并将所有路由使用";"分隔
    routes_str = "; ".join(["->".join([str(segment) for segment in route]) for route in value['候选路由']])
    # 在UD字典中更新路由字符串
    UD[key]['候选路由字符串'] = routes_str

    # 更新demand_df DataFrame
    # 遍历demand_df，为每个需求对添加对应的候选路由字符串
demand_df['候选路由'] = demand_df.apply(lambda row: UD[(row['需求对起点'], row['需求对终点'])]['候选路由字符串'],
                                        axis=1)

# 将更新后的DataFrame写回到原始的Excel文件中，可以选择写入一个新的sheet
with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    demand_df.to_excel(writer, sheet_name='需求信息更新', index=False)

theta = {}
for ud in UD.keys():
    for route in UD[ud]['候选路由']:
        for path in L.keys():
            if path in route:
                theta[(ud, tuple(route), path)] = 1
            else:
                theta[(ud, tuple(route), path)] = 0

lsfin = {}
# 遍历L字典
for key, value in L.items():
    # 提取'线路终点'和'线路类型'
    end_point, path_type = key[1], value['path_type']
    # 创建新键，如果不存在，则初始化为空列表
    if end_point in S.keys():
        new_key = (end_point, path_type)
        if new_key not in lsfin:
            lsfin[new_key] = []
        # 向对应的列表中添加当前线路（键）
        lsfin[new_key].append(key)
    else:
        continue

# 输出结果查看
print(f"进港场地功能")
for key, value in lsfin.items():
    print(f"{key}: {value}")

lsfout = {}
for key, value in L.items():
    # 提取'线路起点'和'线路类型'
    start_point, path_type = key[0], value['path_type']
    if start_point in S.keys():
        # 创建新键，如果不存在，则初始化为空列表
        new_key = (start_point, path_type)
        if new_key not in lsfout:
            lsfout[new_key] = []
        # 向对应的列表中添加当前线路（键）
        lsfout[new_key].append(key)
    else:
        continue

# 输出结果查看
print(f"出港场地功能")
for key, value in lsfout.items():
    print(f"{key}: {value}")

# 权重
alpha = 0.7
beta = 0.3

# 创建模型
model = Model("SFAP")

# 决策变量
x = {s: {stype: model.addVar(vtype=GRB.BINARY, name=f'x_{s}_{stype}') for stype in value['sort_type']} for s, value in
     S.items()}
y = {key: model.addVars(range(len(value['候选路由'])), vtype=GRB.BINARY, name=f"y_{key}") for key, value in UD.items()}
ly = model.addVars(L.keys(), vtype=GRB.BINARY, name="ly")
lq = model.addVars(L.keys(), vtype=GRB.CONTINUOUS, name="lq")
lv = model.addVars(L.keys(), vtype=GRB.INTEGER, name="lv")

# 约束条件
# 路径选择唯一性约束
model.addConstrs((sum(y[key].values()) == 1 for key in UD.keys()), name="path_unique")

# 路径选择与线路启用之间的耦合关系约束
model.addConstrs((ly[path] * quicksum(
    theta[ud, tuple(route), path] for ud in UD.keys() for idx, route in enumerate(UD[ud]['候选路由'])) >= quicksum(
    theta[ud, tuple(route), path] * y[ud][idx] for ud in UD.keys() for idx, route in enumerate(UD[ud]['候选路由'])) for
                  path in L.keys()),
                 name="route_path_coupling"
                 )

# 线路启用与场地功能决策之间的耦合关系约束
model.addConstrs((x[s][f] * len(lsfin.get((s, f), [])) >= sum(
    ly[path] for path in L.keys() if path[1] == s and L[path]['path_type'] == f) for s in S for f in S[s]['sort_type']),
                 name="path_in_sort_function"
                 )

model.addConstrs((x[s][f] * len(lsfout.get((s, f), [])) >= sum(
    ly[path] for path in L.keys() if path[0] == s and L[path]['path_type'] == f) for s in S for f in S[s]['sort_type']),
                 name="path_out_sort_function"
                 )

# 线路货量的计算
model.addConstrs(
    (lq[path] == quicksum(
        theta[ud, tuple(route), path] * y[ud][idx] * UD[ud]['demand'] for ud in UD.keys() for idx, route in
        enumerate(UD[ud]['候选路由'])) for path in L.keys()), name="path_volume"
)

# 线路车辆数的计算
model.addConstrs((L[path]['max_capacity'] * lv[path] >= lq[path] for path in L.keys()), name="path_vehicles")

# 分拣场地产能限制
model.addConstrs((quicksum(lq[path] for path in L.keys() if path[0] == s) <= S[s]['max_volume'] for s in S),
                 name="sort_capacity")

# 分拣场地承接流向数限制
model.addConstrs((quicksum(ly[(s1, s)] for (s1, s) in lsfin.get((s, f), [])) + quicksum(
    ly[(s, s2)] for (s, s2) in lsfout.get((s, f), [])) <=
                  S[s]['max_path'][S[s]['sort_type'].index(f)]
                  for s in S.keys() for f in S[s]['sort_type']),
                 name="flow_limit"
                 )

# 目标函数
model.setObjective(
    alpha * quicksum(L[path]['trans_cost'] * lv[path] for path in L.keys()) +
    beta * quicksum(S[s]['opre_cost'] * lq[path] for s in S.keys() for path in L.keys() if path[0] == s),
    GRB.MINIMIZE
)

# 输出解决方案
try:
    model.optimize()
    if model.status == GRB.INF_OR_UNBD or model.status == GRB.INFEASIBLE:
        print('Model is infeasible. Computing IIS.')
        model.computeIIS()  # 计算不可行约束集
        model.write("model.ilp")  # 将不可行约束集写入文件
    else:
        for var in model.getVars():
            if var.X > 0:
                print(f"{var.VarName}: {var.X}")

except GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))
