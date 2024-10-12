import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from math import radians, cos, sin, asin, sqrt
from itertools import permutations

def geodistance(lng1,lat1,lng2,lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance=2*asin(sqrt(a))*6371*1000
    distance=round(distance/1000,3)
    return distance


class model:
    def __init__(self,path):
        self.dc = pd.read_excel(path,sheet_name='配送中心')[['配送中心','经度','纬度']].set_index('配送中心')
        self.packages = pd.read_excel(path,sheet_name='流量')[['路由','输入节点','输出节点','包裹体积']].set_index('路由')
        self.vehicle = pd.read_excel(path,sheet_name='车型')[['车型','长度','体积',	'成本',	'数量']].set_index('车型')
        self.package_threshold = 5
        self.transit_cost = 20
        self._convert_coordinates('A')
        self.dc_list = list(self.dc.index)
        self.routs = self._get_R()
        self.lines = [''.join(line) for line in permutations(self.dc_list,2)]
        env = gp.Env(params={"OutputFlag": 1})
        self.m = gp.Model(env=env)
    


    def _convert_coordinates(self,origin):
        lng = self.dc.loc[origin,'经度']
        lat = self.dc.loc[origin,'纬度']
        for index,x in self.dc.iterrows():
            self.dc.loc[index,'x'] = geodistance(lng,0,x['经度'],0)
            self.dc.loc[index,'y'] = geodistance(0,lat,0,x['纬度'])
        self.distance = pd.DataFrame()
        for index1,x in self.dc.iterrows():
            for index2,y in self.dc.iterrows():
                self.distance.loc[index1,index2] = geodistance(x['经度'],x['纬度'],y['经度'],y['纬度'])

    def _get_R(self)->dict:
        # 获取候选路由集合
        ls = self.dc_list.copy()
        for index in self.dc_list:
            if index in list(self.packages['输入节点']) or index in list(self.packages['输出节点']):
                ls.remove(index)
        temp = []
        for length in range(len(ls)):
            temp += permutations(ls,length+1)
        routs = {}
        for index,x in self.packages.iterrows():
            begin = x['输入节点']
            end = x['输出节点']
            routs[begin+end] = begin+end
            for line in temp:
                attr = ''.join(tuple([begin]+list(line)+[end]))
                routs[attr] = begin+end
        return routs

    def set_variables(self):
        # 建立变量
        ## y:0-1变量,候选路由r是否使用
        self.y = {}
        for rout in self.routs.keys():
            self.y[rout] = self.m.addVar(vtype=GRB.BINARY ,name = f'y_{rout}')
        ## p:0-1变量,线路l上车型v是否使用
        ## q:整数变量，线路l上车型v车辆数
        self.p = {}
        self.q = {}
        for line in self.lines:
            for vehicle in self.vehicle.index:
                self.p[line,vehicle] = self.m.addVar(vtype=GRB.BINARY  ,name = f'p_{line}_{vehicle}')
                self.q[line,vehicle] = self.m.addVar(vtype=GRB.INTEGER ,name = f'q_{line}_{vehicle}')
        ## z: 0-1变量,线路l是否开通
        self.z = {}
        for line in self.lines:
            self.z[line] = self.m.addVar(vtype=GRB.BINARY  ,name = f'z_{line}')

    def set_object(self):
        # 设定目标
        # 第一项：车辆成本
        cost1 = gp.LinExpr()
        for (line,vehicle) in self.q.keys():
            cost1 += self.q[line,vehicle] * self.vehicle.loc[vehicle,'成本'] * self.distance.loc[line[0],line[1]]
        # 第二项：转运成本
        cost2 = gp.LinExpr()
        for rout in self.y.keys():
            cost2 += self.y[rout] * self.transit_cost * len(rout) * self.packages.loc[self.routs[rout],'包裹体积']
        self.m.setObjective(cost1+cost2, GRB.MINIMIZE)

    def set_constraint(self):
        self.constraint = {}
        ## 约束1：路由唯一性约束
        con1 = {}
        for rout in self.routs.keys():
            if self.routs[rout] in con1.keys():
                con1[self.routs[rout]] += self.y[rout]
            else:
                con1[self.routs[rout]] = self.y[rout]
        for attr in con1.keys():
            self.constraint['c1_'+attr] = self.m.addConstr(con1[attr]==1, name='c1_'+attr)
        ## 约束2：车辆载容约束
        con2_1 = {}
        con2_2 = {}
        for (line,vehicle) in self.q.keys():
            if line in con2_1.keys():
                con2_1[line] += self.q[line,vehicle] * self.vehicle.loc[vehicle,'体积']
            else:
                con2_1[line] = self.q[line,vehicle] * self.vehicle.loc[vehicle,'体积']
        for line in self.lines:
            for rout in self.routs.keys():
                if line in rout:
                    if line in con2_2.keys():
                        con2_2[line] += self.y[rout] * self.packages.loc[self.routs[rout],'包裹体积']
                    else:
                        con2_2[line] = self.y[rout] * self.packages.loc[self.routs[rout],'包裹体积']
        for line in con2_1.keys():
            if line in con2_2.keys():
                self.constraint['c2_'+line] = self.m.addConstr(con2_1[line] >= con2_2[line], name='c2_'+line)
            # else:
            #     print('----',line)
        ## 约束3：车辆数量约束
        for (line,vehicle) in self.p.keys():
            self.constraint['C3'+line+vehicle] = self.m.addConstr(self.p[line,vehicle]*self.vehicle.loc[vehicle,'数量'] >= self.q[line,vehicle], name='C3'+line+vehicle)
        ## 约束4：车型唯一约束
        con4 = {}
        for (line,vehicle) in self.p.keys():
            if line in con4.keys():
                con4[line] += self.p[line,vehicle]
            else:
                con4[line] = self.p[line,vehicle]
        for line in con4.keys():
            self.constraint['c4'+line] = self.m.addConstr(con4[line] == self.z[line], name='c4'+line)
        ## 约束5：线路与路由的逻辑关系约束，其含义为只有当线路l的状态为开通时，线路上才允许有货量流动
        con5 = {}
        for line in self.lines:
            for rout in self.routs.keys():
                if line in rout:
                    if line in con5.keys():
                        con5[line] += self.y[rout] * self.packages.loc[self.routs[rout],'包裹体积']
                    else:
                        con5[line] = self.y[rout] * self.packages.loc[self.routs[rout],'包裹体积']
        M = 999999
        for line in con5.keys():
            self.constraint['c5'+line] = self.m.addConstr(self.z[line]*M >= con5[line], name='c5'+line)
        ## 约束6：为线路开通货量约束，其含义为只有当线路l的状态为开通时，线路上的货量应该大于其开通标准。
        for line in con5.keys():
            self.constraint['c6'+line] = self.m.addConstr(con5[line] >= self.z[line]*self.package_threshold*self.distance.loc[line[0],line[1]], name='c6'+line)
        ## 约束7：路由与线路的对应关系
        for line in self.lines:
            for rout in self.routs.keys():
                if line in rout:
                    self.constraint['c7'+line+'_'+rout] = self.m.addConstr(self.z[line] >= self.y[rout], name='c7'+line+'_'+rout)
        
    def solve_model(self):
        self.m.Params.MIPGap = 0.01
        self.m.Params.TimeLimit = 60
        self.m.Params.Heuristics = 1
        self.m.Params.Method = 2
        self.m.Params.MIPFocus = 1
        self.m.Params.NonConvex = 2
        self.m.write('model.lp')
        self.m.optimize()
        self.status = self.m.status
        self.solution = pd.DataFrame()
        if self.m.status not in {2, 7, 8, 9, 10, 13, 15}:
            print(f'模型求解失败,模型状态:{self.m.status}')
            self.m.computeIIS()
            self.m.write("model_file.ilp")
        else:
            print(f'模型存在可行解,模型状态:{self.m.status}')
            self.obj = self.m.objVal
            print('目标值为：',self.obj)
            # 存储解
            self.solution_y = self._getSol(self.y)
            self.solution_p = self._getSol(self.p)
            self.solution_q = self._getSol(self.q)
            self.solution_z = self._getSol(self.z)
            # 第一项：车辆成本
            cost1 = 0
            for (line,vehicle) in self.q.keys():
                cost1 += self.q[line,vehicle].x * self.vehicle.loc[vehicle,'成本'] * self.distance.loc[line[0],line[1]]
            # 第二项：转运成本
            cost2 = 0
            for rout in self.y.keys():
                cost2 += self.y[rout].x * self.transit_cost * len(rout) * self.packages.loc[self.routs[rout],'包裹体积']
            print('运输成本：',cost1)
            print('转运成本：',cost2)
    
    def _getSol(self,s)->dict:
        solution = {}
        for line in s.keys():
            solution[line] = s[line].x
        return solution

def print_sol(sol:dict):
    for line in sol.keys():
        if sol[line] != 0.0 and sol[line] != -0.0:
            print(f'{line}:{sol[line]}')