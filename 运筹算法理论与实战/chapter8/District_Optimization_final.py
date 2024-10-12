import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum
import warnings

warnings.filterwarnings('ignore')
# from jsf import *

class StationModel(object):
    def __init__(self,station_name,district_num,object_param = 0.5,\
            balance_param = 0.6,income_param = 1.0, time_param = 1.0, weight_param = 1.0):
        aois = pd.read_csv('data/%s_aois.csv' % station_name, encoding='utf-8')
        edges = pd.read_csv('data/%s_edges.csv' % station_name, encoding = 'utf-8')
        # aois['lng','lat','clt_num','dlv_num','clt_income','dlv_income','clt_weight','dlv_weight','clt_time','dlv_time','dist_to_station'] = \
        #    aois['lng','lat','clt_num','dlv_num','clt_income','dlv_income','clt_weight','dlv_weight','clt_time','dlv_time','dist_to_station'].astype(float)
        aois = aois.astype(float)
        edges = edges.astype(int)

        income_score = []
        time_score = []
        weight_score = []
        edge_num = 0
        from_index = []
        to_index = []
        edge_index = []

        for i in range(len(aois)):
            # 边的信息
            from_index.append([])
            to_index.append([])
        for i in range(len(aois)):
            for j in range(1,edges.shape[1]):
                if edges.iloc[i,j] != -1:
                    from_index[edges.loc[i,'aoi_id']].append(edge_num)
                    to_index[edges.iloc[i,j]].append(edge_num)
                    edge_index.append([edges.loc[i,'aoi_id'],edges.iloc[i,j]])
                    edge_num += 1
            # 节点的权值
            clt_num = aois.loc[i,'clt_num']
            dlv_num = aois.loc[i,'dlv_num']
            # 归一化
            income_score.append(0.01*(clt_num * aois.loc[i,'clt_income'] + dlv_num * aois.loc[i,'dlv_income']))
            time_score.append(0.00003*(clt_num * aois.loc[i,'clt_time'] + dlv_num * aois.loc[i,'dlv_time']))
            weight_score.append(0.0005*(clt_num * aois.loc[i,'clt_weight'] + dlv_num * aois.loc[i,'dlv_weight']))
            #income_score.append((clt_num * aois.loc[i,'clt_income'] + dlv_num * aois.loc[i,'dlv_income']))
            #time_score.append((clt_num * aois.loc[i,'clt_time'] + dlv_num * aois.loc[i,'dlv_time']))
            #weight_score.append((clt_num * aois.loc[i,'clt_weight'] + dlv_num * aois.loc[i,'dlv_weight']))

        self.aoi_data = aois
        self.edge_data = edges
        self.aoi_num = len(aois)
        self.dist_num = district_num
        self.edge_num = edge_num

        self.balance_param = balance_param # 要求不同路区之间收入差距足够小
        # self.object_param = object_param   # 两个目标函数的权重
        self.income_param = income_param   # 收入系数，默认为1
        self.time_param = time_param       # 时间系数
        # self.weight_param = weight_param   # 重量系数

        self.from_index = from_index
        self.to_index = to_index
        self.edge_index = edge_index
        
        self.income_score = income_score
        self.time_score = time_score
        self.weight_score = weight_score

    def optimizer(self,init_solution = []):
        # aux_vars
        big_M = 200

        # Create a new model
        model = gp.Model("mip1")

        # Create variables
        X  = [] # AOI和路区的对应关系
        R  = [] # 是否为第一个承接水流的AOI
        Y  = [] # 边的两端是否在某一路区内
        F  = [] # 起点AOI->终点AOI的水流量
        F0 = [] # 由水塔至各AOI的水流量

        for i in range(self.aoi_num):
            tmp1 = [model.addVar(vtype = GRB.BINARY, name = 'x_'+str(i)+'_'+str(k)) for k in range(self.dist_num)]
            X.append(tmp1)
            tmp2 = [model.addVar(vtype = GRB.BINARY, name = 'r_'+str(i)+'_'+str(k)) for k in range(self.dist_num)]
            R.append(tmp2)
            F0.append(model.addVar(vtype = GRB.CONTINUOUS, lb = 0.0, ub = self.aoi_num, name = 'F0_'+str(i)))

        for i in range(self.aoi_num):
            for j in range(1,self.edge_data.shape[1]):
                if self.edge_data.iloc[i,j] != -1:
                    F.append(model.addVar(vtype = GRB.CONTINUOUS, lb = 0.0, ub = self.aoi_num - 1,
                        name = 'F_'+str(self.edge_data.loc[i,'aoi_id'])+'_'+str(self.edge_data.iloc[i,j])))
                    tmp3 = [model.addVar(vtype = GRB.BINARY,
                        name = 'y_'+str(self.edge_data.loc[i,'aoi_id'])+'_'+str(self.edge_data.iloc[i,j])+'_'+str(k))
                        for k in range(self.dist_num)]
                    Y.append(tmp3)
                else:
                    break

        # # 路区内的最大最小经纬度
        # D1_min = [model.addVar(vtype = GRB.CONTINUOUS, lb = 73, ub = 136, name = 'd1_'+str(k)+'_min') for k in range(self.dist_num)]
        # D1_max = [model.addVar(vtype = GRB.CONTINUOUS, lb = 73, ub = 136, name = 'd1_'+str(k)+'_max') for k in range(self.dist_num)]
        # D2_min = [model.addVar(vtype = GRB.CONTINUOUS, lb = 4,  ub = 53,  name = 'd2_'+str(k)+'_min') for k in range(self.dist_num)]
        # D2_max = [model.addVar(vtype = GRB.CONTINUOUS, lb = 4,  ub = 53,  name = 'd2_'+str(k)+'_max') for k in range(self.dist_num)]
        # # 各路区经度范围/纬度范围较大值
        # D_max = [model.addVar(vtype = GRB.CONTINUOUS,  lb = 0.0, ub = 63, name = 'd_'+str(k)+'_max') for k in range(self.dist_num)]
        # 最高最低总分
        Q_max = model.addVar(vtype = GRB.CONTINUOUS, lb = 0.0, ub = 100, name = 'q_max')
        # Q_min = model.addVar(vtype = GRB.CONTINUOUS, lb = 0.0, ub = 100, name = 'q_min')
        # 最高最低各项评分
        Q_income_max = model.addVar(vtype = GRB.CONTINUOUS, lb = 0.0, ub = 100, name = 'q_income_max')
        Q_income_min = model.addVar(vtype = GRB.CONTINUOUS, lb = 0.0, ub = 100, name = 'q_income_min')
        # Q_time_max = model.addVar(vtype = GRB.CONTINUOUS, name = 'q_time_max')
        # Q_time_min = model.addVar(vtype = GRB.CONTINUOUS, name = 'q_time_min')
        # Q_weight_max = model.addVar(vtype = GRB.CONTINUOUS, name = 'q_weight_max')
        # Q_weight_min = model.addVar(vtype = GRB.CONTINUOUS, name = 'q_weight_min')

        # Set objective
        # model.setObjective(self.object_param * (Q_max) + (1-self.object_param) * quicksum(D_max), GRB.MINIMIZE)
        model.setObjective(Q_max, GRB.MINIMIZE)

        # Add constraint:
        for i in range(self.aoi_num):
            # 2.AOI全覆盖约束
            model.addConstr(quicksum(X[i][k] for k in range(self.dist_num)) == 1)
            # 6.AOI水量守恒约束
            model.addConstr(quicksum(F[l] for l in self.to_index[i]) + F0[i] - quicksum(F[l] for l in self.from_index[i]) == 1)
            # 11.直接从水塔接水节点约束
            model.addConstr(F0[i] <= quicksum(R[i][k] for k in range(self.dist_num)) * self.aoi_num)
            for k in range(self.dist_num):
                # 12.最早承接水流约束
                model.addConstr(R[i][k] <= X[i][k])
                # 13.最早承接水流的AOI是路区内编号最小的AOI
                model.addConstr(self.aoi_num * R[i][k] <= self.aoi_num + 1 - quicksum(X[j][k] for j in range(i+1)))
                # # 18-21.路区内的最大最小经纬度
                # model.addConstr(D1_min[k] <= self.aoi_data.loc[i,'lng'] + (1-X[i][k]) * big_M)
                # model.addConstr(D2_min[k] <= self.aoi_data.loc[i,'lat'] + (1-X[i][k]) * big_M)
                # model.addConstr(D1_max[k] >= self.aoi_data.loc[i,'lng'] - (1-X[i][k]) * big_M)
                # model.addConstr(D2_max[k] >= self.aoi_data.loc[i,'lat'] - (1-X[i][k]) * big_M)

        for k in range(self.dist_num):
            # 4.最大最小收入路区；时间和重量的平衡暂不考虑
            model.addConstr(Q_income_min <= quicksum(self.income_score[i] * X[i][k] for i in range(self.aoi_num)))
            model.addConstr(Q_income_max >= quicksum(self.income_score[i] * X[i][k] for i in range(self.aoi_num)))
            # model.addConstr(Q_time_min <= quicksum(self.time_score[i]*X[i][k] for i in range(self.aoi_num)) + \
            #                 quicksum(self.aoi_data.loc[i,'dist_to_station']*R[i][k] for i in range(self.aoi_num))/(20/3.6))
            # model.addConstr(Q_time_max >= quicksum(self.time_score[i]*X[i][k] for i in range(self.aoi_num)) + \
            #                 quicksum(self.aoi_data.loc[i,'dist_to_station']*R[i][k] for i in range(self.aoi_num))/(20/3.6))
            # model.addConstr(Q_weight_min <= quicksum(self.weight_score[i]*X[i][k] for i in range(self.aoi_num)))
            # model.addConstr(Q_weight_max >= quicksum(self.weight_score[i]*X[i][k] for i in range(self.aoi_num)))

            # 10.水流覆盖路区约束
            model.addConstr(quicksum(R[i][k] for i in range(self.aoi_num)) == 1)
            # 14-15.最大最小评分路区
            model.addConstr(Q_max >= self.income_param*quicksum(self.income_score[i]*X[i][k] for i in range(self.aoi_num)) \
                - self.time_param*quicksum(self.time_score[i]*X[i][k] for i in range(self.aoi_num)) \
                - 0.00003*self.time_param*quicksum(self.aoi_data.loc[i,'dist_to_station']*R[i][k] for i in range(self.aoi_num))*2/(20/3.6))
            # model.addConstr(Q_max >= self.income_param*quicksum(self.income_score[i]*X[i][k] for i in range(self.aoi_num)) \
            #     - self.time_param*quicksum(self.time_score[i]*X[i][k] for i in range(self.aoi_num)) \
            #     - 0.00003*self.time_param*quicksum(self.aoi_data.loc[i,'dist_to_station']*R[i][k] for i in range(self.aoi_num))*2/(20/3.6) \
            #     - self.weight_param*quicksum(self.weight_score[i]*X[i][k] for i in range(self.aoi_num)))
            #model.addConstr(Q_min <= self.income_param*quicksum(self.income_score[i]*X[i][k] for i in range(self.aoi_num)) \
            #    - self.time_param*quicksum(self.time_score[i]*X[i][k] for i in range(self.aoi_num)) \
            #    - 0.00003*self.time_param*quicksum(self.aoi_data.loc[i,'dist_to_station']*R[i][k] for i in range(self.aoi_num))*2/(20/3.6) \
            #    - self.weight_param*quicksum(self.weight_score[i]*X[i][k] for i in range(self.aoi_num)))
            # # 16-17.路区内的最大单维度距离
            # model.addConstr(D_max[k] >= D1_max[k] - D1_min[k])
            # model.addConstr(D_max[k] >= D2_max[k] - D2_min[k])

        for l in range(self.edge_num):
            # 7.水量只在同一路区内分配
            model.addConstr(F[l] <= self.aoi_num * quicksum(Y[l][k] for k in range(self.dist_num)))
            # 8.变量Y与变量X匹配
            for k in range(self.dist_num):
                model.addConstr(X[self.edge_index[l][0]][k] + X[self.edge_index[l][1]][k] - 2 * Y[l][k] >= 0)
                model.addConstr(X[self.edge_index[l][0]][k] + X[self.edge_index[l][1]][k] - 2 * Y[l][k] <= 1)
            if self.edge_index[l][0] < self.edge_index[l][1]:
                # 9.y_ij=y_ji约束，用于降低解空间
                for s in self.from_index[self.edge_index[l][1]]:
                    if self.edge_index[s][1] == self.edge_index[l][0]:
                        for k in range(self.dist_num):
                            model.addConstr(Y[l][k] == Y[s][k])
                        break

        # 3.路区收入差距约束
        model.addConstr(Q_income_min >= self.balance_param * Q_income_max)
        # 5.总水量约束
        model.addConstr(quicksum(F0) == self.aoi_num)

        # Model param and init solution settings
        model.setParam("Cuts", 2)
        model.setParam("Heuristics",0.4)
        model.setParam("MIPFocus",1)
        model.setParam("TimeLimit",1800)
        model.setParam("MIPGap",0.0001)
        if len(init_solution) > 0:
            Q_max.Start = 100
            # Q_min.Start = -100
            for k in range(self.dist_num):
                for i in range(len(init_solution)):
                    X[i][k].Start = (init_solution[i]==k)
                for l in range(self.edge_num):
                    Y[l][k].Start = init_solution[self.edge_index[l][0]] == k and init_solution[self.edge_index[l][1]] == k

        # Optimize model
        model.optimize()

        for v in model.getVars():
            var = v.VarName.split('_')
            if v.X == 1 and var[0] == 'x':
                print(var[0]+'_'+var[1]+': '+var[2])
            elif var[0] == 'q' or var[0] == 'd':
                print('%s: %f' % (v.VarName, v.X))

        print('Obj: %g' % model.ObjVal)

if __name__ == '__main__':
    site_model = StationModel('test40_station', 4, 0.5, 0.6, 1, 1, 1)
    init_solution4_2 = [0,0,1,1]
    init_solution40_5 = [0,0,0,1,1,2,2,2,0,0,0,1,1,2,2,2,0,0,1,1,1,2,3,3,4,4,4,4,4,3,3,3,4,4,4,4,4,3,3,3]
    init_solution40_4 = [1,1,1,0,0,2,2,2,1,1,1,0,0,0,2,2,1,1,0,0,0,0,2,2,1,3,3,3,3,3,3,2,3,3,3,3,3,2,2,2]
    init_solution40_3 = [0,0,0,1,1,1,2,2,0,0,0,1,1,1,2,2,0,0,0,1,1,1,2,2,0,0,0,1,1,2,2,2,0,0,0,1,1,2,2,2]
    init_solution500_20 = []
    for i in range(20):
        for j in range(25):
            init_solution500_20.append(np.floor(i/5)*5 + np.floor(j/5))
        
    site_model.optimizer(init_solution40_4)
