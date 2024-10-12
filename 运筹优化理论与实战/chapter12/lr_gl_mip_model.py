from pyscipopt import Model
from pyscipopt import quicksum
import pandas as pd
import numpy as np

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)


def build_order_data():
    data_df = pd.read_csv('/图书仓组单数据.csv')
    # 筛选需要的字段
    data_df = data_df[['outbound_no', 'goods_no', 'zone_no', 'aisles_no',
                       'cell_no', 'aisles_seq', 'locate_qty', 'goods_volume', 'goods_weight']]
    data_df.to_excel('输入数据.xlsx',index=False)
    # 将单位转化为升
    data_df['goods_volume'] = data_df['goods_volume'] / 1000000
    data_df['volume'] = data_df['goods_volume'] * data_df['locate_qty']
    data_df['weight'] = data_df['goods_weight'] * data_df['locate_qty']
    # 添加订单明细id
    data_df['id'] = range(0, data_df.shape[0])
    # 添加订单 id
    zone_list = data_df['outbound_no'].unique()
    zone_df = pd.DataFrame({'outbound_no': zone_list, 'outbound_id': range(0, len(zone_list))})
    data_df = pd.merge(data_df, zone_df, on='outbound_no', how='left')
    # 添加储区 id
    zone_list = data_df['zone_no'].unique()
    zone_df = pd.DataFrame({'zone_no': zone_list, 'zone_id': range(0, len(zone_list))})
    data_df = pd.merge(data_df, zone_df, on='zone_no', how='left')
    # 添加 巷道id
    aisles_list = data_df['aisles_no'].unique()
    aisles_df = pd.DataFrame({'aisles_no': aisles_list, 'aisles_id': range(0, len(aisles_list))})
    data_df = pd.merge(data_df, aisles_df, on='aisles_no', how='left')
    # 添加 储位id
    cell_list = data_df['cell_no'].unique()
    cell_df = pd.DataFrame({'cell_no': cell_list, 'cell_id': range(0, len(cell_list))})
    data_df = pd.merge(data_df, cell_df, on='cell_no', how='left')

    return data_df


def solve(data_df: pd.DataFrame, max_batch_count=6, max_order_count=40,
          max_qty=280, max_volume=2000, max_weight=1000):
    model = Model()
    model.setParam("display/freq",1)
    order_d_count = data_df.shape[0]
    order_count = data_df['outbound_id'].nunique()
    cell_count = data_df['cell_id'].nunique()
    aisles_count = data_df['aisles_id'].nunique()
    zone_count = data_df['zone_id'].nunique()

    print(f"待分配数据，订单数={order_count},订单明细数={order_d_count}，最大集合单数={max_batch_count}")
    x = {}  # 0-1变量，订单明细i是否分配到集合单b
    u = {}  # 0-1变量，订单j是否分配到集合单b
    v = {}  # 0-1变量，集合单b中是否包含储位s
    p = {}  # 0-1变量，集合单b中是否包含巷道l
    q = {}  # 0-1变量，集合单b中是否包含储区a
    l_min = {}  # 连续变量，集合单b中最小巷道顺序编号，l_min^b∈L；
    l_max = {}  # 连续变量，集合单b中最大巷道顺序编号，l_max^b∈L
    for i in range(order_d_count):
        for b in range(0,max_batch_count + 1):
            x[i, b] = model.addVar(f"x_{i}_{b}", "B")
    for j in range(order_count):
        for b in range(0,max_batch_count + 1):
            u[j, b] = model.addVar(f"x_{j}_{b}", "B")
    for s in range(cell_count):
        for b in range(1, max_batch_count + 1):
            v[s, b] = model.addVar(f"x_{s}_{b}", "B")
    for l in range(aisles_count):
        for b in range(1, max_batch_count + 1):
            p[l, b] = model.addVar(f"x_{l}_{b}", "B")
    for a in range(zone_count):
        for b in range(1, max_batch_count + 1):
            q[a, b] = model.addVar(f"x_{a}_{b}", "B")
    for b in range(1, max_batch_count + 1):
        l_min[b] = model.addVar(f"l_min_{b}", "I")  # 集合单最小巷道顺序
        l_max[b] = model.addVar(f"l_max__{b}", "I")  # 集合单最大巷道顺序

    # 每个订单明细i只能分配到一个集合单b下
    for i in range(order_d_count):
        model.addCons(quicksum(x[i, b] for b in range(0,max_batch_count + 1)) == 1)

    for outbound_id, gp_data in data_df.groupby(by='outbound_id'):
        id_list = gp_data['id'].tolist()
        # 同一订单的订单明细，必须分配到相同集合单
        for i in range(1, len(id_list)):
            for b in range(0, max_batch_count + 1):
                model.addCons(x[id_list[i - 1], b] == x[id_list[i], b])
        # 订单明细被分配到集合单b，该明细对应的订单被分配到集合单b
        outbound_id = int(outbound_id)
        for detail_id in id_list:
            for b in range(0, max_batch_count + 1):
                model.addCons(u[outbound_id, b] >= x[detail_id, b])

    max_aisle = data_df.aisles_seq.max()
    min_aisle = data_df.aisles_seq.min()

    qty_arr = data_df.locate_qty.values  # 件数
    volume_arr = data_df.volume.values  # 体积
    weight_arr = data_df.weight.values  # 重量
    aisles_seq_arr = data_df.aisles_seq.values  # 巷道顺序

    for b in range(1, max_batch_count + 1):
        # 集合单最大订单数约束
        model.addCons(quicksum(u[i, b] for i in range(order_count)) <= max_order_count)
        # 集合单最大件数约束
        model.addCons(quicksum(qty_arr[i] * x[i, b] for i in range(order_d_count)) <= max_qty)
        # 集合单最大体积
        model.addCons(quicksum(volume_arr[i] * x[i, b] for i in range(order_d_count)) <= max_volume)
        # 集合单最大重量
        model.addCons(quicksum(weight_arr[i] * x[i, b] for i in range(order_d_count)) <= max_weight)
        # 最大巷道顺序，必须大于等于最小巷道顺序
        model.addCons(l_max[b] >= l_min[b])
        # 集合单巷道顺序范围
        for i in range(order_d_count):
            model.addCons(l_max[b] >= x[i, b] * aisles_seq_arr[i])
            model.addCons(l_min[b] <= x[i, b] * aisles_seq_arr[i] + (1 - x[i, b]) * max_aisle)
            model.addCons(l_max[b] <= max_aisle)
            model.addCons(l_min[b] >= min_aisle)


    for row in data_df.itertuples():
        detail_id = row.id
        cell_id = row.cell_id
        aisles_id = row.aisles_id
        zone_id = row.zone_id
        for b in range(1, max_batch_count + 1):
            # 集合单b有效储位
            model.addCons(v[cell_id, b] >= x[detail_id, b])
            # 集合单b有效巷道
            model.addCons(p[aisles_id, b] >= x[detail_id, b])
            # 集合单b有效储区
            model.addCons(q[zone_id, b] >= x[detail_id, b])

    obj_cell_count = quicksum(v[s, b] for s in range(cell_count) for b in range(1, max_batch_count + 1))
    obj_aisles_count = quicksum(p[s, b] for s in range(aisles_count) for b in range(1, max_batch_count + 1))
    obj_zone_count = quicksum(q[s, b] for s in range(zone_count) for b in range(1, max_batch_count + 1))
    obj_aisles_diff = quicksum(l_max[b] - l_min[b] + 1 for b in range(1, max_batch_count + 1))
    lbd1, lbd2, lbd3, lbd4 = (1, 1, 1, 1)
    # lbd1, lbd2, lbd3, lbd4 = (0, 0, 1, 0)

    M = cell_count * lbd1 + aisles_count * lbd2 + (max_aisle - min_aisle + 1) * lbd3 + zone_count * lbd4 + 500
    print(f"储位数={cell_count},巷道数={aisles_count},储区数={zone_count},最大跨巷道={max_aisle - min_aisle + 1},M={M}")
    psh = quicksum(u[i, 0] for i in range(order_count))
    obj = lbd1 * obj_cell_count + lbd2 * obj_aisles_count + lbd3 * obj_aisles_diff + lbd4 * obj_zone_count + M * psh

    model.setObjective(obj, "minimize")
    model.setParam("limits/time", 1800)
    model.setParam("lp/threads", 8)
    model.optimize()

    order_d_r = [0] * order_d_count
    for i in range(order_d_count):
        for j in range(max_batch_count + 1):
            if model.getVal(x[i, j]) > 0.1:
                order_d_r[i] = j
    data_df['batch_no'] = order_d_r
    obj = model.getObjVal()
    print("obj=", obj)
    order_r = [0] * order_count
    for i in range(order_count):
        for j in range(max_batch_count + 1):
            if model.getVal(u[i, j]) > 0.1:
                order_r[i] = j
    print("订单分配结果")
    print(order_r)
    return data_df


def show_result(data_df: pd.DataFrame):
    result_df = (data_df[data_df.batch_no >= 0].groupby(by='batch_no').agg(
        {
            'aisles_seq': ['min', 'max'],
            'locate_qty': 'sum',
            'outbound_no': pd.Series.nunique,
            'cell_id': pd.Series.nunique,
            'aisles_id': pd.Series.nunique,
            'zone_id': pd.Series.nunique,
            'volume': 'sum',
            'weight': 'sum'
        }).reset_index())
    result_df.columns = ['batch_no', 'min_aisles_seq', 'max_aisles_seq', 'qty', 'order_count',
                         'cell_count', 'aisles_count', 'zone_count', 'volume', 'weight']
    result_df['cross_aisles'] = result_df['max_aisles_seq'] - result_df['min_aisles_seq'] + 1
    # 实际生成集合单数量
    dispatch_batch_count = result_df.loc[result_df.batch_no > 0].shape[0]
    # 实际分配订单数量
    dispatch_order_count = result_df.loc[result_df.batch_no > 0, 'order_count'].sum()
    # 未分配订单数
    failed_order_count = result_df.loc[result_df.batch_no == 0, 'order_count'].sum()
    print(f"实际生成集合单数量={dispatch_batch_count},分配订单数={dispatch_order_count},未分配订单数={failed_order_count}")

    cross_aisle = result_df.loc[result_df.batch_no > 0,'cross_aisles'].sum()
    cell_count = result_df.loc[result_df.batch_no > 0,'cell_count'].sum()
    aisles_count = result_df.loc[result_df.batch_no > 0, 'aisles_count'].sum()
    zone_count = result_df.loc[result_df.batch_no > 0, 'zone_count'].sum()

    print(f"实际生成集合单，总跨巷道数={cross_aisle},储区数={zone_count},巷道数={aisles_count},储位数={cell_count}")
    print(result_df.head(20))
    result_df.to_excel('计算结果.xlsx',index=False)


if __name__ == '__main__':
    outbound_data = build_order_data()
    result_df = solve(outbound_data)
    show_result(result_df)
