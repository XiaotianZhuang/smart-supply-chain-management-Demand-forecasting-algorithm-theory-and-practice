# -*- coding:utf-8 -*-
import pandas as pd
from gurobipy import *
from pandas.errors import SettingWithCopyWarning
import warnings

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

'''
供应链教材仓网规划章节案例
'''

if __name__ == '__main__':
    baseline = False  # 是否现状还原

    # 1. 读取数据
    input_file = '案例数据.xlsx'
    excel_data = pd.read_excel(input_file, sheet_name=None)
    node_df = excel_data['节点']
    product_df = excel_data['商品']
    order_df = excel_data['订单']
    transport_df = excel_data['线路']

    # 数据加工
    U = set(node_df[node_df['仓库类型'] != 'FDC']['仓库名称'])
    F = set(node_df[node_df['仓库类型'] == 'FDC']['仓库名称'])
    WY = set(node_df[node_df['类型'] == '必选']['仓库名称'])
    WN = set(node_df[node_df['类型'] == '不选']['仓库名称'])
    CU = set(order_df['目的商店'])

    # 商品信息
    pw = product_df.set_index('商品名称')['重量（kg）'].to_dict()
    pv = product_df.set_index('商品名称')['体积（m3）'].to_dict()

    # 连接关系相关变量
    EI = transport_df.groupby('目的地').agg({'始发地': set})['始发地'].to_dict()
    EO = transport_df.groupby('始发地').agg({'目的地': set})['目的地'].to_dict()
    EP = order_df.groupby('目的商店').agg({'商品名称': set})['商品名称'].to_dict()
    EP.update({i: set(product_df['商品名称']) for i in U | F})
    ET = transport_df.groupby('目的地').agg({'运输方式': set})['运输方式'].to_dict()
    LP = {f'{i}_{j}': EP.get(j) for i, out in EO.items() for j in out}
    LPT = {f'{i}_{j}_{p}': set(transport_df[(transport_df['始发地'] == i) & (transport_df['目的地'] == j)]['运输方式']) for i, out in EO.items() for j in out for p in LP.get(f'{i}_{j}')}
    CUI = {f'{j}_{p}_{t}': set(transport_df[(transport_df['目的地'] == j) & (transport_df['运输方式'] == t)]['始发地']) for j in CU for p in EP[j] for t in ET[j]}

    # 需求
    order_df['目的-商品-运输'] = order_df['目的商店'] + '_' + order_df['商品名称'] + '_' + order_df['运输类型']
    d = order_df.groupby('目的-商品-运输').agg({'需求量（件）': 'sum'})['需求量（件）'].to_dict()

    # 成本
    transport_df['始发-目的-运输'] = transport_df['始发地'] + '_' + transport_df['目的地'] + '_' + transport_df['运输方式']
    tcp = transport_df.set_index('始发-目的-运输')['成本（元/kg）'].to_dict()
    node_out_cost = node_df.set_index('仓库名称')['出库成本（元/kg）'].to_dict()
    ocp = {(node, p): cost * w for node, cost in node_out_cost.items() for p, w in pw.items()}
    node_in_cost = node_df.set_index('仓库名称')['入库成本（元/kg）'].to_dict()
    icp = {(node, p): cost * w for node, cost in node_in_cost.items() for p, w in pw.items()}
    rcp = node_df.set_index('仓库名称')['租金（元/天·㎡）'].to_dict()
    theta = node_df.set_index('仓库名称')['坪效'].to_dict()
    sf = node_df.set_index('仓库名称')['安全库存天数'].to_dict()

    beta = 5  # 选仓数量
    re = 10  # 补货周期
    M = sum(order_df['需求量（件）']) + 1
    L = min(order_df['需求量（件）'])

    # 2. 建模
    model = Model("Network_planning")

    x_var = {}  # 仓库节点选择变量
    y_var = {}  # 上下游连接变量
    z_var = {}  # 线路流量变量
    Q_var = {}  # 仓库出库件数
    W_var = {}  # 仓库出库重量
    V_var = {}  # 仓库出库体积
    tc_var = {}  # 运输费用
    oc_var = {}  # 出库费用
    ic_var = {}  # 入库费用
    rc_var = {}  # 仓租费用
    for i in U | F:
        x_var[i] = model.addVar(vtype='B', name=f'x_{i}')
    # 上下游连接变量
    for j, origin in EI.items():
        for i in origin:
            y_var[f'{i}_{j}'] = model.addVar(vtype='B', name=f'y_{i}_{j}')
    # 上下游商品运输方式流量变量
    for j, origin in EI.items():
        for i in origin:
            for p in LP.get(f'{i}_{j}'):
                for t in LPT.get(f'{i}_{j}_{p}'):
                    z_var[f'{i}_{j}_{p}_{t}'] = model.addVar(lb=0, vtype='C', name=f'ypt_{i}_{j}_{p}_{t}')

    # 流量相关变量
    for i in U | F:
        Q_var[i] = model.addVar(lb=0, vtype='C', name=f'Q_{i}')
        W_var[i] = model.addVar(lb=0, vtype='C', name=f'W_{i}')
        V_var[i] = model.addVar(lb=0, vtype='C', name=f'V_{i}')

    # 成本相关变量
    for j, origin in EI.items():
        for i in origin:
            for p in LP.get(f'{i}_{j}'):
                for t in LPT.get(f'{i}_{j}_{p}'):
                    tc_var[f'{i}_{j}_{p}_{t}'] = model.addVar(lb=0, vtype='C', name=f'tc_{i}_{j}_{p}_{t}')
    TC_var = model.addVar(lb=0, vtype='C', name='TC')
    for i in U | F:
        oc_var[i] = model.addVar(lb=0, vtype='C', name=f'oc_{i}')
        ic_var[i] = model.addVar(lb=0, vtype='C', name=f'ic_{i}')
        rc_var[i] = model.addVar(lb=0, vtype='C', name=f'rc_{i}')
    OC_var = model.addVar(lb=0, vtype='C', name='OC')
    IC_var = model.addVar(lb=0, vtype='C', name='IC')
    RC_var = model.addVar(lb=0, vtype='C', name='RC')

    # 2.2 添加约束
    # 约束1：覆盖关系约束
    # 建约束addConstrs时遍历dict不要用for key, value in dict.items(),生成generate的index会有value信息，导致gurobi变量名字超长
    # 如果仓库没有被选择，则不能覆盖下游
    model.addConstrs((z_var[f'{i}_{j}_{p}_{t}'] - M * x_var[i] <= 0 for i in EO.keys() for j in EO[i] for p in LP.get(f'{i}_{j}') for t in LPT.get(f'{i}_{j}_{p}')), name='select_con1')
    # 如果一个仓库被选择，则至少覆盖一个
    model.addConstrs((quicksum([z_var[f'{i}_{j}_{p}_{t}'] for j in EO[i] for p in LP.get(f'{i}_{j}') for t in LPT.get(f'{i}_{j}_{p}')]) - L * x_var[i] >= 0 for i in EO.keys()), name='select_con2')

    # 约束2：连接关系约束
    # 如果连接没有被选择，则不能有流量
    model.addConstrs((z_var[f'{i}_{j}_{p}_{t}'] - M * y_var[f'{i}_{j}'] <= 0 for i in EO.keys() for j in EO[i] for p in LP.get(f'{i}_{j}') for t in LPT.get(f'{i}_{j}_{p}')), name='cover_con1')
    # 如果连接被选择，则必须有流量
    model.addConstrs((quicksum([z_var[f'{i}_{j}_{p}_{t}'] for p in LP.get(f'{i}_{j}') for t in LPT.get(f'{i}_{j}_{p}')]) - L * y_var[f'{i}_{j}'] >= 0 for i in EO.keys() for j in EO[i]), name='cover_con2')

    # 约束3：流量计算
    model.addConstrs((Q_var[i] - quicksum([z_var[f'{i}_{j}_{p}_{t}'] for j in EO[i] for p in LP.get(f'{i}_{j}') for t in LPT.get(f'{i}_{j}_{p}')]) == 0 for i in EO.keys()), name='node_piece_con')
    model.addConstrs((W_var[i] - quicksum([z_var[f'{i}_{j}_{p}_{t}'] * pw[p] for j in EO[i] for p in LP.get(f'{i}_{j}') for t in LPT.get(f'{i}_{j}_{p}')]) == 0 for i in EO.keys()), name='node_weight_con')
    model.addConstrs((V_var[i] - quicksum([z_var[f'{i}_{j}_{p}_{t}'] * pv[p] for j in EO[i] for p in LP.get(f'{i}_{j}') for t in LPT.get(f'{i}_{j}_{p}')]) == 0 for i in EO.keys()), name='node_volume_con')

    # 约束4：流量平衡约束
    model.addConstrs(
        (quicksum([z_var[f'{i}_{j}_{p}_{t}'] for j in EO.get(i) if p in LP.get(f'{i}_{j}', {}) for t in LPT.get(f'{i}_{j}_{p}')]) - quicksum([z_var[f'{j}_{i}_{p}_{t}'] for j in EI.get(i) for t in LPT.get(f'{j}_{i}_{p}')]) == 0 for i in EO.keys() if i in EI.keys() for p in EP[i]),
        name='flow_balance_con')

    # # 约束5：客户运输类型-产品满足约束
    model.addConstrs((quicksum([z_var[f'{i}_{j}_{p}_{t}'] for i in CUI[f'{j}_{p}_{t}']]) == d.get(f'{j}_{p}_{t}', 0) for j in CU for p in EP.get(j) for t in ET[j]), name='demand_con')

    # 约束6：单一上游约束
    model.addConstrs((quicksum([y_var[f'{i}_{j}'] for i in EI[j]]) == x_var[j] for j in EI.keys() if j in U | F), name='node_single_source_con')
    model.addConstrs((quicksum([y_var[f'{i}_{j}'] for i in EI[j]]) == 1 for j in EI.keys() if j in CU), name='customer_single_source_con')

    if baseline:
        # 现状还原逻辑 需要 关闭必选不选仓约束 与 选仓数量约束
        current_node = set(order_df['始发仓库'])
        beta = len(current_node)
        current_cover_df = order_df[['始发仓库', '目的商店']].drop_duplicates()
        current_cover_df['od'] = order_df['始发仓库'] + '_' + order_df['目的商店']
        current_cover = set(current_cover_df['od'])
        model.addConstr(quicksum([x_var[i] for i in F]) == beta, name='fdc_num_con')
        model.addConstrs((y_var[c] == 1 for c in current_cover), name='current_cover_con')
    else:
        # 约束7：必选仓不选仓约束
        model.addConstrs((x_var[i] == 1 for i in WY), name='node_select_con')
        model.addConstrs((x_var[i] == 0 for i in WN), name='node_exclude_con')

        # 约束8：选仓数量约束
        model.addConstr(quicksum([x_var[i] for i in F]) == beta, name='fdc_num_con')

    # 2.3 添加目标函数
    # 成本1: 线路运输成本
    model.addConstrs((tc_var[f'{i}_{j}_{p}_{t}'] - tcp[f'{i}_{j}_{t}'] * z_var[f'{i}_{j}_{p}_{t}'] == 0 for i in EO.keys() for j in EO[i] for p in LP.get(f'{i}_{j}') for t in LPT.get(f'{i}_{j}_{p}')), name='transport_cost_con')
    model.addConstr(TC_var - quicksum(tc_var.values()) == 0, name='total_transport_cost_con')
    # 成本2:出入库成本
    model.addConstrs((oc_var[i] - quicksum([z_var[f'{i}_{j}_{p}_{t}'] * ocp[(i, p)] for j in EO.get(i, {}) for p in LP.get(f'{i}_{j}') for t in LPT.get(f'{i}_{j}_{p}')]) == 0 for i in U | F), name='outbound_cost')
    model.addConstr(OC_var - quicksum(oc_var.values()) == 0, name='total_outbound_cost_con')
    model.addConstrs((ic_var[j] - quicksum([z_var[f'{i}_{j}_{p}_{t}'] * icp[(j, p)] for i in EI.get(j, {}) for p in LP.get(f'{i}_{j}') for t in LPT.get(f'{i}_{j}_{p}')]) == 0 for j in U | F), name='inbound_cost')
    model.addConstr(IC_var - quicksum(ic_var.values()) == 0, name='total_inbound_cost_con')
    # 成本3:仓租成本
    model.addConstrs((rc_var[i] - quicksum([z_var[f'{i}_{j}_{p}_{t}'] * rcp[i] * (0.5 * re + sf[i]) / theta[i] for j in EO.get(i, {}) for p in LP.get(f'{i}_{j}') for t in LPT.get(f'{i}_{j}_{p}')]) == 0 for i in U | F), name='rent_cost')
    model.addConstr(RC_var - quicksum(rc_var.values()) == 0, name='total_rent_cost_con')

    model.setObjective(TC_var + OC_var + IC_var + RC_var, sense=GRB.MINIMIZE)

    # 2.4 模型求解
    model.optimize()
    model.write('model.lp')

    # 2.5 整理结果
    if model.status == 2:
        total_result = {'方案总成本': model.ObjVal, '总运输成本': TC_var.x, '总出库成本': OC_var.x, '总入库成本': IC_var.x, '总仓租成本': RC_var.x}
        print(total_result)
        node_result = [[i, oc_var[i].x, ic_var[i].x, rc_var[i].x, Q_var[i].x, W_var[i].x, V_var[i].x] for i in U | F if x_var[i].x > 0.5]
        print(f'选择仓库及流量汇总为{node_result}')
        cover_result = [[i, j] for j, origin in EI.items() for i in origin if y_var[f"{i}_{j}"].x > 0.5]
        print(f'覆盖关系为{cover_result}')
        flow_result = [[i, j, p, t, z_var[f"{i}_{j}_{p}_{t}"].x, tc_var[f"{i}_{j}_{p}_{t}"].x] for j, origin in EI.items() for i in origin for p in LP.get(f'{i}_{j}') for t in LPT.get(f'{i}_{j}_{p}') if z_var[f"{i}_{j}_{p}_{t}"].x > 0.0]
        print(f'运输流量为{flow_result}')

        # 结果写入Excel
        node_result_df = pd.DataFrame(node_result, columns=['仓库名称', '出库成本', '入库成本', '仓租成本', '发出件数', '发出重量', '发出体积'])
        cover_result_df = pd.DataFrame(cover_result, columns=['始发地', '目的地'])
        flow_result_df = pd.DataFrame(flow_result, columns=['始发地', '目的地', '商品名称', '运输方式', '流量', '运输成本'])
        time_df = transport_df[['始发地', '目的地', '运输时效（h）']].drop_duplicates()
        flow_result_df = pd.merge(flow_result_df, time_df, on=['始发地', '目的地'], how='left')
        # 计算时效
        average_time_df = flow_result_df[['流量', '运输时效（h）']]
        average_time_df['加权时效'] = average_time_df.apply(lambda x: x['流量'] * x['运输时效（h）'], axis=1)
        average_time = sum(average_time_df['加权时效']) / sum(average_time_df['流量'])
        print(f'加权时效为{average_time}')
        summary_result = {'FDC数量': [beta], '方案总成本': [model.ObjVal], '总运输成本': [TC_var.x], '总出库成本': [OC_var.x], '总入库成本': [IC_var.x], '总仓租成本': [RC_var.x], '加权时效（h）': [average_time]}

        print('计算完成')

        with pd.ExcelWriter(f"计算结果_{('现状' if baseline else str(beta)+'仓')}方案.xlsx") as writer:
            pd.DataFrame(summary_result).to_excel(writer, sheet_name='汇总信息', index=False)
            node_result_df.to_excel(writer, sheet_name='仓库信息', index=False)
            cover_result_df.to_excel(writer, sheet_name='覆盖关系', index=False)
            flow_result_df.to_excel(writer, sheet_name='流量信息', index=False)
    else:
        model.computeIIS()
        model.write('model.ilp')
        raise Exception('模型不可解')
