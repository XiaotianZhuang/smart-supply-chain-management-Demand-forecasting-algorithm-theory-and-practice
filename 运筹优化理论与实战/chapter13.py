import time
import logging
from ortools.linear_solver import pywraplp

global version


class BalanceModelOrtools:
    modelFileName = "inventoryBalanceModel_python.txt"

    @staticmethod
    def model_ab_ortools(wh_dict: dict, obj_expr=None, opt_obj=None, opt_sol=None):
        # 存储配入仓和配入仓的列表
        inbound_wh = [w for w in wh_dict.keys() if wh_dict[w]['transDemand'] > 0 and len(wh_dict[w]['vlt_list'].keys()) > 0]
        outbound_wh = [w for w in wh_dict.keys() if wh_dict[w]['transDemand'] < 0 and len(wh_dict[w]['vlt_list'].keys()) > 0]

        for i in outbound_wh:
            if wh_dict[i]['outbound_demand'] == 0:
                logging.info("仓库{}的outbound_whw为0".format(i))
        for j in inbound_wh:
            if wh_dict[j]['inbound_demand'] == 0:
                logging.info("仓库{}的inbound_whw为0".format(j))
        # 大M
        big_m_z_new = max(sum([wh_dict[i]['outbound_demand'] for i in outbound_wh]),
                          sum([wh_dict[j]['inbound_demand'] for j in inbound_wh]))
        # big_m_s = max(wh_dict[j]['stock'] for j in inbound_wh)
        # solver = pywraplp.Solver.CreateSolver('CBC')
        solver = pywraplp.Solver("MipProgram", pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
        big_m_y = max([wh_dict[j]['inbound_demand'] for j in inbound_wh])
        big_m_s = max([wh_dict[j]['stock'] for j in inbound_wh]) + big_m_y
        # 定义变量
        infinity = solver.infinity()
        x = {outbound_wh[i]: {inbound_wh[j]: solver.IntVar(0, infinity, 'x_{}_{}'.format(i, j)) for j in range(len(inbound_wh))} for i in range(len(outbound_wh))}
        p_in = {inbound_wh[j_1]: {inbound_wh[j_2]: solver.NumVar(0, infinity, 'p_in_{}_{}'.format(j_1, j_2)) for j_2 in range(len(inbound_wh)) if j_2 > j_1} for j_1 in
                range(len(inbound_wh))}
        p_out = {outbound_wh[i_1]: {outbound_wh[i_2]: solver.NumVar(0, infinity, 'p_out_{}_{}'.format(i_1, i_2)) for i_2 in range(len(outbound_wh)) if i_2 > i_1} for i_1 in
                 range(len(outbound_wh))}
        z_min = {inbound_wh[j]: solver.IntVar(0, 1, 'z_min_{}'.format(j)) for j in range(len(inbound_wh))}
        y = {outbound_wh[i]: {inbound_wh[j]: solver.IntVar(0, 1, 'y_{}_{}'.format(i, j)) for j in range(len(inbound_wh))} for i in range(len(outbound_wh))}
        z_in = solver.IntVar(0, 1, 'z-in')
        z_out = solver.IntVar(0, 1, 'z-out')

        # 定义约束
        # 最小保有量是否满足
        for j in inbound_wh:
            solver.Add(wh_dict[j]['stock'] + solver.Sum([x[i][j] for i in outbound_wh]) - wh_dict[j]['minHold'] >= -z_min[j] * big_m_s)

        # 均衡需求不需要超量满足
        for i in outbound_wh:
            solver.Add(solver.Sum([x[i][j] for j in inbound_wh]) <= wh_dict[i]['outbound_demand'])

        # 可配出库存上限
        for j in inbound_wh:
            solver.Add(solver.Sum(x[i][j] for i in outbound_wh) <= wh_dict[j]['inbound_demand'])

        # 是否调拨
        for i in outbound_wh:
            for j in inbound_wh:
                solver.Add(x[i][j] <= big_m_y * y[i][j])

        # 满足均衡需求所覆盖日期的预测需求
        solver.Add(sum(wh_dict[i]['outbound_demand'] for i in outbound_wh) - solver.Sum(
            x[i][j] for i in outbound_wh for j in inbound_wh) <= big_m_z_new * z_out)
        solver.Add(sum(wh_dict[j]['inbound_demand'] for j in inbound_wh) - solver.Sum(
            x[i][j] for i in outbound_wh for j in inbound_wh) <= big_m_z_new * z_in)

        solver.Add(z_out + z_in <= 1)

        # 所有仓均衡需求满足率的波动性（方差）
        for j_1 in range(len(inbound_wh)):
            for j_2 in range(len(inbound_wh)):
                if j_1 < j_2:
                    solver.Add((wh_dict[inbound_wh[j_1]]['stock'] + solver.Sum(x[i][inbound_wh[j_1]] for i in outbound_wh)) / (
                            wh_dict[inbound_wh[j_1]]['stock'] + wh_dict[inbound_wh[j_1]]['inbound_demand']) - (
                                       wh_dict[inbound_wh[j_2]]['stock'] + solver.Sum(x[i][inbound_wh[j_2]] for i in outbound_wh)) / (
                                       wh_dict[inbound_wh[j_2]]['stock'] + wh_dict[inbound_wh[j_2]]['inbound_demand']) <= p_in[inbound_wh[j_1]][inbound_wh[j_2]])
                    solver.Add((wh_dict[inbound_wh[j_1]]['stock'] + solver.Sum(x[i][inbound_wh[j_1]] for i in outbound_wh)) / (
                            wh_dict[inbound_wh[j_1]]['stock'] + wh_dict[inbound_wh[j_1]]['inbound_demand']) - (
                                       wh_dict[inbound_wh[j_2]]['stock'] + solver.Sum(x[i][inbound_wh[j_2]] for i in outbound_wh)) / (
                                       wh_dict[inbound_wh[j_2]]['stock'] + wh_dict[inbound_wh[j_2]]['inbound_demand']) >= - p_in[inbound_wh[j_1]][inbound_wh[j_2]])

        # 配出仓剩余库存方差
        spot_stock = []
        for i in outbound_wh:
            spot_stock.append(wh_dict[i]['stock'])
        if len(spot_stock) != 0:
            max_spot = max(max(spot_stock), 1)
        else:
            max_spot = 1
        for i_1 in range(len(outbound_wh)):
            for i_2 in range(len(outbound_wh)):
                if i_1 < i_2:
                    solver.Add((wh_dict[outbound_wh[i_1]]['stock'] - solver.Sum(x[outbound_wh[i_1]][j] for j in inbound_wh)) / max_spot -
                               (wh_dict[outbound_wh[i_2]]['stock'] - solver.Sum(x[outbound_wh[i_2]][j] for j in inbound_wh)) / max_spot <=
                               p_out[outbound_wh[i_1]][outbound_wh[i_2]])
                    solver.Add((wh_dict[outbound_wh[i_1]]['stock'] - solver.Sum(x[outbound_wh[i_1]][j] for j in inbound_wh)) / max_spot -
                               (wh_dict[outbound_wh[i_2]]['stock'] - solver.Sum(x[outbound_wh[i_2]][j] for j in inbound_wh)) / max_spot >=
                               -p_out[outbound_wh[i_1]][outbound_wh[i_2]])

        # 定义目标函数
        # Objective function
        v_max = []
        for j in inbound_wh:
            v_max.append(max([wh_dict[j]['vlt_list'][x] for x in wh_dict[j]['vlt_list'].keys() if x in outbound_wh]))
        v_max = max(v_max)
        # 最低保有量惩罚
        penalty_minhold = solver.Sum(z_min[j] for j in inbound_wh)
        # 均衡目标满足率方差
        if len(inbound_wh) == 1:
            satisfactory_rate_variation_obj = 0
        else:
            satisfactory_rate_variation = []
            count = 1.0
            for j_1 in range(len(inbound_wh)):
                for j_2 in range(len(inbound_wh)):
                    if j_2 > j_1:
                        # count += 1.0
                        satisfactory_rate_variation.append(p_in[inbound_wh[j_1]][inbound_wh[j_2]] * count)
            # satisfactory_rate_variation_obj = solver.Sum(p_in[inbound_wh[j_1]][inbound_wh[j_2]] for j_1 in range(len(inbound_wh)) for j_2 in range(len(inbound_wh))
            #                                              if j_2 > j_1) / (len(inbound_wh) ** 2 - len(inbound_wh))
            satisfactory_rate_variation_obj = solver.Sum(satisfactory_rate_variation) / (len(inbound_wh) ** 2 - len(inbound_wh))
        # 运输成本
        transshipment_cost = []
        count = 1.0
        for j in range(len(inbound_wh)):
            for i in range(len(outbound_wh)):
                # count += 1.0 + i * j
                transshipment_cost.append((y[outbound_wh[i]][inbound_wh[j]] * (1 + v_max + wh_dict[inbound_wh[j]]['vlt_list'][outbound_wh[i]]) * count) / (1 + 2 * v_max))
        # transshipment_cost_obj = solver.Sum((y[i][j] * (1 + v_max + wh_dict[j]['vlt_list'][i]))/(
        #         1 + 2 * v_max) for i in outbound_wh for j in inbound_wh)
        transshipment_cost_obj = solver.Sum(transshipment_cost)
        # 配出仓剩余库存
        count = 1.0
        remain_inv = []
        for i_1 in range(len(outbound_wh)):
            for i_2 in range(len(outbound_wh)):
                if i_2 > i_1:
                    # count += 1.0
                    if len(outbound_wh) == 1:
                        remain_inv.append(0)
                    else:
                        remain_inv.append((p_out[outbound_wh[i_1]][outbound_wh[i_2]] * 2 * count) / ((1 + 2 * v_max) * (len(outbound_wh) ** 2 - len(outbound_wh))))
        # remain_inv_var = solver.Sum(p_out[outbound_wh[i_1]][outbound_wh[i_2]] * 2/((1+2 * v_max)*(len(outbound_wh)**2 - len(outbound_wh)))
        #                             for i_1 in range(len(outbound_wh)) for i_2 in range(len(outbound_wh)) if i_2 > i_1)
        remain_inv_var = solver.Sum(remain_inv)
        obj = 100 * penalty_minhold + 200 * satisfactory_rate_variation_obj + transshipment_cost_obj + remain_inv_var

        solver.Minimize(obj)
        solver.set_time_limit(time_limit_milliseconds=60000)
        start_time = time.time()
        status = solver.Solve()
        end_time = time.time()
        # fo = open(BalanceModelOrtools.modelFileName, "w")
        # fo.write(solver.ExportModelAsLpFormat(False))
        # fo.close()

        # Extract optimal solutions

        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            # logging.info('求解的问题状态为{}, 用时={}'.format('optimal' if status == 0 else 'feasible', (end_time - start_time)))
            opt_sol = {
                'x': {i: {j: x[i][j].solution_value() for j in inbound_wh} for i in outbound_wh},
                'z_in': z_in.solution_value(),
                'z_out': z_out.solution_value(),
                'p_in': {inbound_wh[j_1]: {inbound_wh[j_2]: p_in[inbound_wh[j_1]][inbound_wh[j_2]].solution_value() for j_2 in range(len(inbound_wh)) if j_2 > j_1} for j_1 in
                         range(len(inbound_wh))},
                'p_out': {outbound_wh[i_1]: {outbound_wh[i_2]: p_out[outbound_wh[i_1]][outbound_wh[i_2]].solution_value() for i_2 in range(len(outbound_wh)) if i_2 > i_1} for i_1
                          in
                          range(len(outbound_wh))}
            }
            logging.info('x_ij:')
            # print('x_ij:')
            for i in range(len(outbound_wh)):
                for j in range(len(inbound_wh)):
                    if x[outbound_wh[i]][inbound_wh[j]].solution_value() > 0:
                        logging.info('{} -> {} : {}'.format(outbound_wh[i], inbound_wh[j], x[outbound_wh[i]][inbound_wh[j]].solution_value()))
                        # print('{} -> {} : {}'.format(outbound_wh[i], inbound_wh[j], x[outbound_wh[i]][inbound_wh[j]].solution_value()))
            logging.info('obj: {}'.format(solver.Objective().Value()))
            # print('obj: {}'.format(solver.Objective().Value()))
            return opt_sol, solver.Objective().Value()
        else:
            return None, None


if __name__ == '__main__':
    input_dict = {'110007045': {'fcstSalesList': [0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0],
                                'inbound_demand': 0,
                                'minHold': 10,
                                'outbound_demand': 12,
                                'stock': 22,
                                'transDemand': -12,
                                'vlt_list': {'110000000': 5,
                                             '110029825': 5,
                                             '110016790': 5,
                                             '110008814': 5,
                                             '118070589': 7,
                                             '110008980': 5,
                                             '110008912': 5},
                                'warehouseId': '110007045'},
                  '110008814': {'fcstSalesList': [0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0],
                                'inbound_demand': 0,
                                'minHold': 5,
                                'outbound_demand': 12,
                                'stock': 17,
                                'transDemand': -12,
                                'vlt_list': {'110000000': 4,
                                             '110029825': 5,
                                             '110016790': 4,
                                             '118070589': 6,
                                             '110008980': 4,
                                             '110007045': 4,
                                             '110008912': 6},
                                'warehouseId': '110008814'},
                  '110008912': {'fcstSalesList': [0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0],
                                'inbound_demand': 3,
                                'minHold': 5,
                                'outbound_demand': 0,
                                'stock': 2,
                                'transDemand': 3,
                                'vlt_list': {'110000000': 5,
                                             '110029825': 4,
                                             '110016790': 5,
                                             '110008814': 4,
                                             '118070589': 6,
                                             '110008980': 4,
                                             '110007045': 5},
                                'warehouseId': '110008912'},
                  '110008980': {'fcstSalesList': [0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0],
                                'inbound_demand': 1,
                                'minHold': 5,
                                'outbound_demand': 0,
                                'stock': 4,
                                'transDemand': 1,
                                'vlt_list': {'110000000': 4,
                                             '110029825': 4,
                                             '110016790': 4,
                                             '110008814': 4,
                                             '118070589': 6,
                                             '110007045': 5,
                                             '110008912': 4},
                                'warehouseId': '110008980'},
                  '110016790': {'fcstSalesList': [0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0],
                                'inbound_demand': 2,
                                'minHold': 5,
                                'outbound_demand': 0,
                                'stock': 3,
                                'transDemand': 2,
                                'vlt_list': {'110000000': 0,
                                             '110029825': 4,
                                             '110008814': 4,
                                             '118070589': 5,
                                             '110008980': 4,
                                             '110007045': 4,
                                             '110008912': 4},
                                'warehouseId': '110016790'},
                  '110029825': {'fcstSalesList': [0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0],
                                'inbound_demand': 2,
                                'minHold': 5,
                                'outbound_demand': 0,
                                'stock': 3,
                                'transDemand': 2,
                                'vlt_list': {'110000000': 4,
                                             '110016790': 4,
                                             '110008814': 5,
                                             '118070589': 5,
                                             '110008980': 4,
                                             '110007045': 5,
                                             '110008912': 5},
                                'warehouseId': '110029825'},
                  '118070589': {'fcstSalesList': [0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0,
                                                  0.0],
                                'inbound_demand': 2,
                                'minHold': 5,
                                'outbound_demand': 0,
                                'stock': 3,
                                'transDemand': 2,
                                'vlt_list': {'110000000': 6,
                                             '110029825': 4,
                                             '110016790': 6,
                                             '110008814': 6,
                                             '110008980': 6,
                                             '110007045': 6,
                                             '110008912': 6},
                                'warehouseId': '118070589'}}

    sol, obj = BalanceModelOrtools.model_ab_ortools(wh_dict=input_dict)
    # opt_obj_dict = {'opt_sat_days': obj}
    # 用字典保存最优解
    sol_dict = {}
    if sol is not None:
        for _i in list(sol['x'].keys()):
            for _j in list(sol['x'][_i].keys()):
                if _j not in sol_dict.keys():
                    sol_dict[_j] = {}
                sol_dict[_j][_i] = sol['x'][_i][_j]

    print(sol_dict)
