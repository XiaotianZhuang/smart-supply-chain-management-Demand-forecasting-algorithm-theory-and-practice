import pandas as pd
import gurobipy as gp
from gurobipy import GRB


class model:
    def __init__(self,path):
        self.price = 5.0
        self.M = 999999
        self.pack_grade = pd.read_excel(path,sheet_name='包裹等级',index_col=0)
        self.car_grade = pd.read_excel(path,sheet_name='整车等级',index_col=0)
        self.order = pd.read_excel(path,sheet_name='订单详情',index_col=0)
        env = gp.Env(params={"OutputFlag": 0})
        self.m = gp.Model(env=env)
    def set_variables(self):
        # 变量z：零担0还是整车1
        # 变量y: 车是否使用
        # 变量x：包裹在哪辆车
        self.x = {}
        self.y = {}
        self.z = {}
        for index,x in self.car_grade.iterrows():
            grade = index
            for id in range(int(x['数量'])):
                id = str(id)
                self.y[grade,id] = self.m.addVar(vtype=GRB.BINARY ,name = 'y_'+grade+id)
        for index,x in self.order.iterrows():
            order = index
            packages = x['包裹等级']
            self.z[order] = self.m.addVar(vtype=GRB.BINARY ,name = 'z_'+order)
            for package_id in range(len(packages)):
                package_id = packages[package_id] + str(package_id)
                for (grade,id) in self.y.keys():
                    self.x[order,package_id,grade,id] = self.m.addVar(vtype=GRB.BINARY ,name = 'x_'+order+package_id+grade+id)
    def set_object(self):
        # 设定目标
        # 第一项：零担成本
        scatter_cost = gp.LinExpr()
        for index,x in self.order.iterrows():
            order = index
            packages = x['包裹等级']
            for package in packages:
                weight = self.pack_grade.loc[package,'重量']
                scatter_cost += self.price * (1 - self.z[order]) * weight
        # 第二项：整车成本
        integral_cost = gp.LinExpr()
        for index,x in self.car_grade.iterrows():
            grade = index
            cost = x['整车成本']
            for id in range(int(x['数量'])):
                id = str(id)
                integral_cost += self.y[grade,id] * cost
        # 惩罚项：是否拆单
        pun1 = {}
        for (order,package_id,grade,id) in self.x.keys():
            if (order,grade,id) in pun1.keys():
                pun1[order,grade,id] += self.x[order,package_id,grade,id]
            else:
                pun1[order,grade,id] = self.x[order,package_id,grade,id]
        temp_v = {}
        for attrs in pun1.keys():
            temp_v[attrs] = self.m.addVar(vtype=GRB.INTEGER )
            self.m.addConstr(temp_v[attrs]==pun1[attrs]**2)
        pun2 = {}
        for (order,grade,id) in pun1.keys():
            if order in pun2.keys():
                pun2[order] += temp_v[order,grade,id]*self.z[order]
            else:
                pun2[order] = temp_v[order,grade,id]*self.z[order]
        pun3 = gp.LinExpr()
        for order in pun2.keys():
            temp = len(self.order.loc[order,'包裹等级'])
            pun3 += self.M * (temp**2*self.z[order] - pun2[order])
        # print(pun2)
        self.m.setObjective(scatter_cost+integral_cost+pun3, GRB.MINIMIZE)
    def set_constraint(self):
        self.constraint = {}
        # # 约束1：选择零担将不消耗整车资源
        # con1 = {}
        # for (order,package_id,grade,id) in self.x.keys():
        #     if order in con1.keys():
        #         con1[order] += self.x[order,package_id,grade,id]
        #     else:
        #         con1[order] = self.x[order,package_id,grade,id]
        # for order in con1.keys():
        #     self.constraint['c0',order] = self.m.addConstr(con1[order]<=self.M*self.z[order], name='c0_'+order)
        # 约束1：一个包裹只能对应一辆车
        con2 = {}
        for (order,package_id,grade,id) in self.x.keys():
            if (order,package_id) in con2.keys():
                con2[order,package_id] += self.x[order,package_id,grade,id]
            else:
                con2[order,package_id] = self.x[order,package_id,grade,id]
        for (order,package_id) in con2.keys():
            self.constraint['c1',order+package_id] = self.m.addConstr(con2[order,package_id]==self.z[order], name='c1_'+order)
        # 约束2：整车的体积限制和承重限制
        con3 = {}
        con4 = {}
        for (order,package_id,grade,id) in self.x.keys():
            if (grade,id) in con3.keys():
                con3[grade,id] += self.x[order,package_id,grade,id] * self.pack_grade.loc[package_id[0],'体积']
                con4[grade,id] += self.x[order,package_id,grade,id] * self.pack_grade.loc[package_id[0],'重量']
            else:
                con3[grade,id] = self.x[order,package_id,grade,id] * self.pack_grade.loc[package_id[0],'体积']
                con4[grade,id] = self.x[order,package_id,grade,id] * self.pack_grade.loc[package_id[0],'重量']
        for (grade,id) in con3.keys():
            self.constraint['c2_v',grade+id] = self.m.addConstr(con3[grade,id]<=self.car_grade.loc[grade,'体积上限']*self.y[grade,id], name='c2_v_'+grade+id)
            self.constraint['c2_w',grade+id] = self.m.addConstr(con4[grade,id]<=self.car_grade.loc[grade,'承重上限']*self.y[grade,id], name='c2_w_'+grade+id)

        
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
            # 存储解
            for order in self.z.keys():
                temp = self.z[order].x
                if temp == 0:
                    self.solution.loc[order,'零担'] = 1
                else:
                    self.solution.loc[order,'零担'] = 0
            for (order,package_id,grade,id) in self.x.keys():
                temp = self.x[order,package_id,grade,id].x
                if temp == 1:
                    if grade+id in list(self.solution.columns):
                        self.solution.loc[order,grade+id] += package_id
                    else:
                        self.solution.loc[order,grade+id] = package_id
                        self.solution = self.solution.fillna('')
            self.obj = self.m.objVal
            print('目标值为：',self.obj)
            # 第一项：零担成本
            scatter_cost = 0
            for index,x in self.order.iterrows():
                order = index
                packages = x['包裹等级']
                for package in packages:
                    weight = self.pack_grade.loc[package,'重量']
                    scatter_cost += self.price * (1 - self.z[order].x) * weight
            # 第二项：整车成本
            integral_cost = 0
            for index,x in self.car_grade.iterrows():
                grade = index
                cost = x['整车成本']
                for id in range(int(x['数量'])):
                    id = str(id)
                    integral_cost += self.y[grade,id].x * cost
            print('总成本为：',scatter_cost+integral_cost)