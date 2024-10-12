import random
from pyscipopt import Model, quicksum
from typing import Optional
import traceback
import pandas as pd
import numpy as np
import datetime
import math
from scipy.stats import norm

M = 99999  # 定义一个非常大的数，用于模型中的大M方法
N = 1e-3  # 定义一个非常小的数，用于避免除以0等数学问题
vlt_default = 5  # 默认的供货周期

# 定义一个库存模型类
class VolModelBySCIPSaleDayRate:
    @staticmethod
    def get_solution(data: pd.DataFrame, rate: dict, last_stock: float, vlt_dict: dict) -> Optional[pd.DataFrame]:
        """
        获取优化模型的解决方案。
        :param data:包含SKU的日销量数据的DataFrame
            字段样例
                datetime 包含每一天的日期，15天的日期就是从 2024-04-01 ~ 2024-04-15
                hist_sales:[1,3,4,...,0,0,0,1,3] 包含历史销量数据
                fcst_sales:[1,3,4,...,0,0,0,1,3] 包含预测销量数据
                sku_id:SKU编码
                store_id:SKU在库仓的编码
                sales_qtty:当天销量
        :param rate: 包含业务相关的配置参数，如满足率、现货率和周转天数
            字段样例
            fulfillRate: 满足率
            stockRate: 现货率
            turnoverValue: 周转
        :param last_stock: 期末库存
        :param vlt_dict: 每个仓库的供货周期（vlt）的字典
        :return:如果成功，返回优化后的DataFrame结果；否则返回None
        """
        # 从数据中获取开始和结束日期
        start_date = data['datetime'].min()
        end_date = data['datetime'].max()
        # 将日期转换为datetime对象
        date_p = datetime.datetime.strptime(str(start_date), '%Y-%m-%d').date()
        date_p1 = datetime.datetime.strptime(str(end_date), '%Y-%m-%d').date()
        # 计算日期范围内的天数
        days = (date_p1 - date_p).days + 1
        # 获取SKU列表和仓库列表
        skuList = sorted(data.sku_id.unique())
        storeList = sorted(data.store_id.unique())
        # 从配置参数中读取满足率、现货率和周转天数
        fillrate = rate['fulfillRate']
        stockrate = rate['stockRate']
        turnoverdays = int(math.ceil(rate['turnoverValue']))

        try:
            model = Model('targetByMinCost')  # 创建模型
            model.hideOutput()  # 隐藏模型输出
            model.setParam("limits/gap", 0.01)  # 设置求解精度
            model.disablePropagation()  # 禁用传播算法优化

            # 定义模型中的变量
            X = {}  # 期末库存
            Y = {}  # 缺货数量
            q = {}  # 补货量
            A = {}  # 缺货指示变量
            B = {}  # 补货指示变量
            S = {}  # 安全库存
            SUB = {}  # 补货上限
            R = {}  # 补货点
            SX = {}  # 安全库存与当前库存的差值
            U1 = {}  # 辅助变量
            U2 = {}
            U3 = {}
            U4 = {}
            num = len(skuList)  # SKU数量
            tmpA = []  # 临时变量数组
            tmpX = []
            tmpB = []
            tmpC = []

            # 创建变量
            nrt = 1  # 补货周期
            if storeList[0] in vlt_dict.keys():  # 检查vlt_dict中是否有对应仓库的vlt值
                vlt = vlt_dict[storeList[0]]
            else:
                vlt = vlt_default

            theta = model.addVar(lb=1.65, ub=8, name='')  # 创建theta变量，用于安全库存计算

            all_score = 0  # 初始化总分数
            demands = []  # 需求列表
            for i in range(num):  # 遍历每个SKU
                sku_sales = data[data.sku_id == skuList[i]].sort_values(by='datetime').reset_index(drop=True)
                X[i, 0] = sum(sku_sales.iloc[: vlt]['sales_qtty']) + vlt * N  # 初始化期初库存

                s_sqrt = math.sqrt(vlt + nrt)  # 安全库存计算中的系数
                r_sqrt = math.sqrt(vlt)  # 补货点计算中的系数
                for t in range(1, days + 1):  # 遍历每天
                    tmp_t = sku_sales.iloc[[t - 1]]
                    _fcst_sales = tmp_t['fcst_sales'].iloc[0]
                    _hist_sales = tmp_t['hist_sales'].iloc[0]
                    safety_stock_days = vlt + int(nrt)
                    mean = np.mean(_fcst_sales[:min(len(_fcst_sales), safety_stock_days)])
                    std = np.std(_hist_sales)
                    S[i, t] = model.addVar(lb=1, name='S_{}_{}'.format(i, t))
                    R[i, t] = model.addVar(lb=1, name='R_{}_{}'.format(i, t))
                    model.addCons(S[i, t] == mean * (vlt + nrt) + theta * s_sqrt * std,
                                  name='con_ss_{}_{}'.format(i, t))
                    SUB[i, t] = mean * (vlt + nrt + 60) #+ theta * s_sqrt * std
                    model.addCons(R[i, t] == mean * (vlt + nrt) + theta * r_sqrt * std,
                                  name='con_rq_{}_{}'.format(i, t))
                    if t == days:
                        X[i, t] = model.addVar(lb=last_stock, name='X_{}_{}'.format(i, t))
                    else:
                        X[i, t] = model.addVar(lb=0, name='X_{}_{}'.format(i, t))
                    Y[i, t] = model.addVar(lb=0, name='Y_{}_{}'.format(i, t))
                    q[i, t] = model.addVar(lb=0, ub=SUB[i, t], name='q_{}_{}'.format(i, t))
                    A[i, t] = model.addVar(vtype='B', name='A_{}_{}'.format(i, t))
                    B[i, t] = model.addVar(vtype='B', name='B_{}_{}'.format(i, t))

                    U1[i, t] = model.addVar(vtype='B', name='U1_{}_{}'.format(i, t))
                    U2[i, t] = model.addVar(vtype='B', name='U2_{}_{}'.format(i, t))
                    U3[i, t] = model.addVar(vtype='B', name='U3_{}_{}'.format(i, t))
                    U4[i, t] = model.addVar(vtype='B', name='U4_{}_{}'.format(i, t))
                    # s-x
                    SX[i, t] = model.addVar(lb=-M, name='SX_{}_{}'.format(i, t))
                    tmpX.append(X[i, t])
                    tmpA.append(A[i, t])
                    tmpB.append(B[i, t])

            for i in range(num):
                sku_sales = data[data.sku_id == skuList[i]].sort_values(by='datetime').reset_index(drop=True)
                for t in range(1, days + 1):
                    demand = sku_sales.iloc[[t - 1]]['sales_qtty'].item()
                    demands.append(str(demand))
                    if t <= vlt:
                        model.addCons(X[i, t] - Y[i, t] == X[i, t - 1] - demand, name='X_1_{}_{}'.format(i, t))

                    else:
                        model.addCons(X[i, t] - Y[i, t] == X[i, t - 1] + q[i, t - vlt] - demand,
                                      name='X_2_{}_{}'.format(i, t))

                    model.addCons(X[i, t] <= M * (1 - A[i, t]), name='X_3_{}_{}'.format(i, t))
                    model.addCons(Y[i, t] <= M * A[i, t], name='X_4_{}_{}'.format(i, t))

                    # on_the_way = []
                    # for j in range(max(1, t - vlt), t):
                    #     on_the_way.append(q[i, j])
                    model.addCons(S[i, t] - X[i, t] <= SX[i, t])
                    model.addCons(0 <= SX[i, t])
                    model.addCons(S[i, t] - X[i, t] >= SX[i, t] - M * (1 - U1[i, t]))
                    model.addCons(0 >= SX[i, t] - M * (1 - U2[i, t]))
                    model.addCons(U1[i, t] + U2[i, t] >= 1)
                    # 判断是否需要补货， 当现货加在途小于补货点库存时，B = 1，将会补货，否则 B = 0 不补货
                    model.addCons(R[i, t] - X[i, t] <= M * B[i, t])
                    model.addCons(R[i, t] - X[i, t] >= M * (B[i, t] - 1) + N)
                    # 确定补货量
                    model.addCons(q[i, t] <= SUB[i, t] * B[i, t])
                    model.addCons(q[i, t] <= SX[i, t])
                    model.addCons(q[i, t] >= SX[i, t] - SUB[i, t] * (1 - B[i, t]))

            # print('约束添加成功')
            # 满足率的约束，加入了订单的权重后的
            # model.addCons(quicksum(tmpC) <= (1 - fillrate) * all_score, name='obj_rate')
            model.addCons(quicksum(tmpA) <= (1 - stockrate) * num * days, name='on_hand_rate')
            objvar = model.addVar(name="objvar")
            model.addCons(objvar >= quicksum(tmpX), name='obj')

            model.setObjective(objvar, 'minimize')
            model.optimize()
            sol = model.getBestSol()
            obj = model.getSolObjVal(sol)
            print('最优的目标值为{}'.format(obj))
            # 保存最终结果
            res = []
            for i in range(num):
                tmp = {}
                tmp['sku_id'] = skuList[i]
                tmp['store_id'] = storeList[i]
                tmp['vlt'] = vlt
                sku_sales = data[data.sku_id == skuList[i]].sort_values(by='datetime').reset_index(drop=True)
                init_stock = sum(sku_sales.iloc[: vlt]['sales_qtty']) + vlt * N
                if model.getStatus() == 'optimal':
                    print(f'sku_list={skuList} store_list={storeList} 现货率最大化模型有最优解')
                tmp['init_stock'] = init_stock
                tmp['theta'] = model.getSolVal(sol, theta)
                # 假设正态分布的均值和标准差
                mu = 0  # 均值
                sigma = 1  # 标准差
                # 创建正态分布对象
                dist = norm(mu, sigma)
                x_value = model.getSolVal(sol, theta)
                # 使用cdf方法计算对应的概率值p
                k_value = dist.cdf(x_value)
                print(k_value)
                tmp['k_best'] = k_value
                s_list = []
                r_list = []
                q_list = []
                q_x_list = []
                is_out_stock_list = []
                end_stock_list = []
                out_stock_num_list = []
                mean_list = []
                std_list = []
                for t in range(1, days + 1):
                    tmp_t = sku_sales.iloc[[t - 1]]
                    _fcst_sales = tmp_t['fcst_sales'].iloc[
                        0]  # list(map(float, str(tmp_t['fcst_sales'].item()).split(',')))
                    _hist_sales = tmp_t['hist_sales'].iloc[
                        0]  # list(map(int, str(tmp_t['hist_sales'].item()).split(',')))
                    safety_stock_days = vlt + int(nrt)
                    mean = int(np.mean(_fcst_sales[:min(len(_fcst_sales), safety_stock_days)]))
                    std = int(np.std(_hist_sales))
                    mean_list.append(str(mean))
                    std_list.append(str(std))
                    s_list.append(str(int(model.getSolVal(sol, S[i, t]))))
                    r_list.append(str(int(model.getSolVal(sol, R[i, t]))))
                    q_list.append(str(int(model.getSolVal(sol, q[i, t]))))
                    q_x_list.append(str(int(model.getSolVal(sol, B[i, t]))))
                    is_out_stock_list.append(str(int(model.getSolVal(sol, A[i, t]))))
                    end_stock_list.append(str(int(model.getSolVal(sol, X[i, t]))))
                    out_stock_num_list.append(str(int(model.getSolVal(sol, Y[i, t]))))
                tmp['demand_list'] = ','.join([str(x) for x in demands])
                tmp['mean_list'] = ','.join(mean_list)
                tmp['std_list'] = ','.join(std_list)
                tmp['q_list'] = ','.join(q_list)
                tmp['q_x_list'] = ','.join(q_x_list)
                tmp['is_out_stock_list'] = ','.join(is_out_stock_list)
                tmp['end_stock_list'] = ','.join(end_stock_list)
                tmp['out_stock_num_list'] = ','.join(out_stock_num_list)
                tmp['s_list'] = ','.join(s_list)
                tmp['r_list'] = ','.join(r_list)
                res.append(tmp)
            df = pd.DataFrame(res)
            df.to_csv('gurobi_model_result1.csv', index=False, encoding='gbk')
            return None
        except Exception as e:
            print('Error code ' + str(traceback.print_exc()) + ': ' + str(e))


if __name__ == '__main__':
    # 随机生成一个输入数据案例
    data = pd.DataFrame({'datetime': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 日期序列（初始为整数，将会被转换为日期）
                         'hist_sales': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 历史销量（初始值，将会被随机生成的序列替换）
                         'sku_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 商品ID（初始值，将会统一更改为'sku1'）
                         'store_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 门店ID（初始值，将会统一更改为'wh1'）
                         'sales_qtty': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 销售数量（初始值，将会被随机生成的序列替换）
                         'fcst_sales': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})  # 预测销量（初始值，将会被随机生成的序列替换）
    # 将'datetime'列的整数序列转换为日期序列，从2024年4月1日开始，每天递增
    data['datetime'] = pd.date_range(start='2024-04-01', end='2024-04-10').strftime('%Y-%m-%d')
    tmp1 = []  # 用于存储随机生成的历史销量
    tmp2 = []  # 本例中未使用，但可用于存储其他随机生成的数据
    for i in range(len(data)):
        tmp1.append([random.randint(1, 100) for _ in range(60)])  # 为每一天生成60个随机数作为历史销量
        tmp2.append([random.randint(1, 100) for _ in range(60)])  # 为每一天生成60个随机数作为预测销量
    data['hist_sales'] = pd.Series(tmp1)  # 更新历史销量列
    data['fcst_sales'] = pd.Series(tmp2)  # 更新预测销量列
    data['sku_id'] = 'sku1'  # 将所有商品ID统一设置为'sku1'
    data['store_id'] = 'wh1'  # 将所有仓库ID统一设置为'wh1'
    data['sales_qtty'] = pd.Series([random.randint(1, 100) for _ in range(10)])  # 随机生成10个销售数量

    rate = {'fulfillRate': 0.95, 'stockRate': 0.99, 'turnoverValue': 20}  # 定义一些参数，如满足率、库存率、周转值
    last_stock = 100  # 上一次的库存量
    vlt_dict = {'13': 3}  # 供应链信息，表示仓库ID为'13'的商品的提前期为3天

    # 调用模型
    VolModel = VolModelBySCIPSaleDayRate()  # 实例化模型对象
    VolModel.get_solution(data, rate, last_stock, vlt_dict)  # 调用模型的求解方法
