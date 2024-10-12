import pandas as pd
# from pyecharts import options as opts
# from pyecharts.charts import Graph
from itertools import combinations
import gurobipy as gp
from gurobipy import GRB
import math


class model:
    def __init__(self,path:str,rout_list:dict):
        self.data = pd.read_excel(path).set_index('配送中心')
        self.data.columns = ['人工流量','分拣机格口数','格口处理货量','人工成本','机器成本']
        self.rout_list = rout_list
        self._get_pack_path()
        # self.draw()
    def _get_pack_path(self):
        # 获取建包路径
        self.packages_path = {}
        self.routs_qty = {}
        for rout in self.rout_list.keys():
            line = list(rout)
            temp_rout = ''.join(map(str,line))
            temp_pack = [line[0] + line[-1]]
            # temp_pack = []
            self.routs_qty[temp_rout] = self.rout_list[rout]
            if len(line)>2:
                temp_pack_split = []
                for index in range(len(line)-2):
                    temp = [list(c) for c in  combinations(line[1:-1], index+1)]
                    for tempsub in temp:
                        temp_pack_split.append(''.join(map(str,tempsub)))
                temp_pack += list(map(lambda x:line[0]+x+line[-1],temp_pack_split))
            self.packages_path[temp_rout] = temp_pack
    def set_model(self):
        env = gp.Env(params={"OutputFlag": 0})
        # 建立模型
        self.m = gp.Model(env=env)
        # 建立变量
        ## 建立x变量: 走货路径 - 建包路径
        self.x = {}
        for rount_path in self.packages_path.keys():
            for pack_path in self.packages_path[rount_path]:
                self.x[rount_path,pack_path] = self.m.addVar(vtype=GRB.BINARY ,name = 'x:' +rount_path+'->'+pack_path)
        ## 建立u变量：建包起点 - 建包终点 - 走货终点：表示在场地i是否将目的地为场地d的小件包裹建包至场地k
        self.u = {}
        for rount_path in self.packages_path.keys():
            end_point = rount_path[-1]
            for pack_path in self.packages_path[rount_path]:
                for point_index in range(len(pack_path)-1):
                    pack_begin = pack_path[point_index]
                    pack_end   = pack_path[point_index+1]
                    if (pack_begin,pack_end,end_point) not in self.u.keys():
                        self.u[pack_begin,pack_end,end_point] = self.m.addVar(vtype=GRB.BINARY ,name = 'u:' +pack_begin+'->'+pack_end+':'+end_point)
        ## 建立y变量：表示在场地i在流向(i,j)是否进行建包操作，i,j∈S；
        self.y = {}
        for rount_path in self.packages_path.keys():
            for begin_point_index in range(len(rount_path)-1):
                for end_point_index in range(begin_point_index+1,len(rount_path)):
                    end_point   = rount_path[end_point_index]
                    begin_point = rount_path[begin_point_index]
                    if (begin_point,end_point) not in self.y.keys():
                        self.y[begin_point,end_point] = self.m.addVar(vtype=GRB.BINARY ,name = 'y:' +begin_point +'->'+ end_point)
        ## 建立y^a变量：表示在场地i在流向(i,j)是否采用人工建包
        self.y_a = {}
        for rount_path in self.packages_path.keys():
            for begin_point_index in range(len(rount_path)-1):
                for end_point_index in range(begin_point_index+1,len(rount_path)):
                    end_point   = rount_path[end_point_index]
                    begin_point = rount_path[begin_point_index]
                    if (begin_point,end_point) not in self.y_a.keys():
                        self.y_a[begin_point,end_point] = self.m.addVar(vtype=GRB.BINARY ,name = 'y_a:' + begin_point +'->'+ end_point)
        ## 建立y^m变量：表示在场地i在流向(i,j)是否采用机器建包
        self.y_m = {}
        for rount_path in self.packages_path.keys():
            for begin_point_index in range(len(rount_path)-1):
                for end_point_index in range(begin_point_index+1,len(rount_path)):
                    end_point   = rount_path[end_point_index]
                    begin_point = rount_path[begin_point_index]
                    if (begin_point,end_point) not in self.y_m.keys():
                        self.y_m[begin_point,end_point] = self.m.addVar(vtype=GRB.BINARY ,name = 'y_m:' + begin_point +'->'+ end_point)
        ## 建立g_m变量：表示当在场地i对流向(i,j)采用机器建包时所需要的格口数量，i∈S,j∈S；
        self.g_m = {}
        for rount_path in self.packages_path.keys():
            for begin_point_index in range(len(rount_path)-1):
                for end_point_index in range(begin_point_index+1,len(rount_path)):
                    end_point   = rount_path[end_point_index]
                    begin_point = rount_path[begin_point_index]
                    if (begin_point,end_point) not in self.g_m.keys():
                        self.g_m[begin_point,end_point] = self.m.addVar(vtype=GRB.INTEGER ,name = 'g_m:' + begin_point +'->'+ end_point)
        ## 建立z_a变量：场地i采用人工建包的总小件包裹量，i∈S
        self.z_a = {}
        for center in self.data.index:
            self.z_a[center] = self.m.addVar(vtype=GRB.INTEGER ,name = 'z_a:' + center)
        ## 建立z_m变量：场地i采用机器建包的总小件包裹量，i∈S
        self.z_m = {}
        for center in self.data.index:
            self.z_m[center] = self.m.addVar(vtype=GRB.INTEGER ,name = 'z_m:' + center)
        ## 建立w_a变量
        self.w_a = {}
        for rount_path in self.packages_path.keys():
            for begin_point_index in range(len(rount_path)-1):
                for end_point_index in range(begin_point_index+1,len(rount_path)):
                    end_point   = rount_path[end_point_index]
                    begin_point = rount_path[begin_point_index]
                    self.w_a[begin_point,end_point] = self.m.addVar(vtype=GRB.INTEGER ,name = 'w_a:' + begin_point+end_point)
        ## 建立w_m变量
        self.w_m = {}
        for rount_path in self.packages_path.keys():
            for begin_point_index in range(len(rount_path)-1):
                for end_point_index in range(begin_point_index+1,len(rount_path)):
                    end_point   = rount_path[end_point_index]
                    begin_point = rount_path[begin_point_index]
                    self.w_m[begin_point,end_point] = self.m.addVar(vtype=GRB.INTEGER ,name = 'w_m:' + begin_point+end_point)
        # 建立约束
        self.constraint = {}
        eps = 0.001
        ## 表示每个流向的小件包裹都需选择且只能选择一条建包路径
        for rount_path in self.packages_path.keys():
            temp = gp.LinExpr()
            for pack_path in self.packages_path[rount_path]:
                temp += self.x[rount_path,pack_path]
            self.constraint['c1_1',rount_path] = self.m.addConstr(temp<=1+eps, name='c1_1_'+rount_path)
            self.constraint['c1_2',rount_path] = self.m.addConstr(temp>=1-eps, name='c1_2_'+rount_path)
        ## 表示对于每个场地而言，目的场地相同的流向需要选择相同的建包路径
        #########################
        temp_1 = {}
        for attrs in self.u.keys():
            pack_begin = list(attrs)[0]
            pack_end = list(attrs)[1]
            end_point = list(attrs)[2]
            if (pack_begin,end_point) in temp_1.keys():
                temp_1[pack_begin,end_point] += self.u[pack_begin,pack_end,end_point]
            else:
                temp_1[pack_begin,end_point] = self.u[pack_begin,pack_end,end_point]
        for attrs in temp_1.keys():
            self.constraint['c2',list(attrs)[0]+list(attrs)[1]] = self.m.addConstr(temp_1[attrs]<=1+eps, name='c2_'+list(attrs)[0]+list(attrs)[1])
        ## 表示当在场地i将目的场地为d的小件包裹建包至场地k （即u_ikd=1） 时，则表示从场地i至场地k的存在建包操作
        for attrs_1 in self.u.keys():
            pack_begin = list(attrs_1)[0]
            pack_end = list(attrs_1)[1]
            if (pack_begin,pack_end) in self.y.keys():
                self.constraint['c3_1',pack_begin+'->'+pack_end] = self.m.addConstr(self.y[pack_begin,pack_end] - self.u[attrs_1] >= -eps, name='c3_1_'+pack_begin+'->'+pack_end)
                self.constraint['c3_2',pack_begin+'->'+pack_end] = self.m.addConstr(self.y[pack_begin,pack_end] - self.u[attrs_1] <= eps, name='c3_2_'+pack_begin+'->'+pack_end)
        ## 式（11.5）表示若对于流向(i,j)，不能建包至场地k（即u_(i,j)=0）时，包含此部分路由的建包路径们不能被选择
        ###########################
        for attrs_1 in self.x.keys():
            end_point = list(attrs_1)[0][-1]
            rout_path = list(attrs_1)[0]
            pack_path = list(attrs_1)[1]
            for attrs_2 in self.u.keys():
                pack_begin,pack_end,end_point_2 = list(attrs_2)[0],list(attrs_2)[1],list(attrs_2)[2]
                path_list = self._get_pack_path_set(pack_begin,pack_end)
                if end_point_2 == end_point and pack_begin+pack_end in pack_path and pack_path in path_list:
                    self.constraint['c4_1',rout_path+pack_begin+pack_end+pack_path] = self.m.addConstr(self.u[attrs_2] - self.x[attrs_1] >= -eps, name='c4_1_'+rout_path+pack_begin+pack_end+pack_path)
                    # self.constraint['c4_2',rout_path+pack_begin+pack_end+pack_path] = self.m.addConstr(self.u[attrs_2] - self.x[attrs_1] <= eps, name='c4_2_'+rout_path+pack_begin+pack_end+pack_path)
        ## 式（11.6）表示对于每个分拣中心的每个流向而言，若建包只能选择人工建包或者选择机器建包，不能二者兼有；
        for attr in self.y.keys():
            if attr in self.y_a.keys() and attr in self.y_m.keys():
                self.constraint['c5_1',attr] = self.m.addConstr(self.y[attr] - self.y_a[attr] - self.y_m[attr] >= -eps, name='c5_1_'+list(attr)[0]+list(attr)[1])
                self.constraint['c5_2',attr] = self.m.addConstr(self.y[attr] - self.y_a[attr] - self.y_m[attr] <= eps, name='c5_2_'+list(attr)[0]+list(attr)[1])
        ## 式（11.7-11.9）表示各个场地需要人工建包操作的每个流向的货量等于各场地所有人工建包流向的小件包裹量之和，其中一个人工建包流向的小件包裹量等于选中的包含该建包路径的所有小件包裹量之和；
        ## 式（11.10-11.12）表示各个场地需要机器建包操作的每个流向的货量等于各场地所有机器建包流向的小件包裹量之和，其中一个机器建包流向的小件包裹量等于选中的包含该建包路径的所有小件包裹量之和；
        self.M = 100000
        for rount_path in self.packages_path.keys():
            for begin_point_index in range(len(rount_path)-1):
                for end_point_index in range(begin_point_index+1,len(rount_path)):
                    end_point   = rount_path[end_point_index]
                    begin_point = rount_path[begin_point_index]
                    temp = gp.LinExpr()
                    rs = self._get_rout_path_set(begin_point,end_point)
                    print(begin_point,end_point,"----------")
                    print(rs)
                    for r in rs:
                        for pack_path in self.packages_path[r]:
                            if begin_point in pack_path and end_point in pack_path:
                                temp += self.x[r,pack_path] * self.routs_qty[r]
                    self.constraint['c6',begin_point+end_point] = self.m.addConstr(temp - self.M*(1-self.y_a[begin_point,end_point]) - (self.w_a[begin_point,end_point]) <= eps, name='c6_'+begin_point+end_point)
                    self.constraint['c7',begin_point+end_point] = self.m.addConstr(temp - (self.w_a[begin_point,end_point]) >= -eps, name='c7_'+begin_point+end_point)

                    self.constraint['c9',begin_point+end_point] = self.m.addConstr(temp - self.M*(1-self.y_m[begin_point,end_point]) - self.w_m[begin_point,end_point] <= eps, name='c9_'+begin_point+end_point)
                    self.constraint['c10',begin_point+end_point] = self.m.addConstr(temp - self.w_m[begin_point,end_point] >= -eps, name='c10_'+begin_point+end_point)
        temp = {}
        for attr in self.w_a.keys():
            i = list(attr)[0]
            j = list(attr)[1]
            if i in temp.keys():
                temp[i] += self.w_a[attr]
            else:
                temp[i] = self.w_a[attr]
        for i in temp.keys():
            self.constraint['c8',i] = self.m.addConstr(self.z_a[i] - temp[i] >= -eps, name='c8_'+i)
            
        temp = {}
        for attr in self.w_m.keys():
            i = list(attr)[0]
            j = list(attr)[1]
            if i in temp.keys():
                temp[i] += self.w_m[attr]
            else:
                temp[i] = self.w_m[attr]
        for i in temp.keys():
            self.constraint['c11',i] = self.m.addConstr(self.z_m[i] - temp[i] >= -eps, name='c11_'+i)
        ## 式（11.13）表示各个分拣中心的人工建包流向数小于或等于现有的人工建包流向数；
        temp = {}
        for attr in self.y_a.keys():
            i = list(attr)[0]
            j = list(attr)[1]
            if i in temp.keys():
                temp[i] += self.y_a[attr] 
            else:
                temp[i] = self.y_a[attr]
        for i in temp.keys():
            self.constraint['c12',i] = self.m.addConstr(self.data.loc[i,'人工流量'] - temp[i] >= -eps, name='c12_'+i)
        ## 式（11.14）表示每个流向货量占用分拣机的隔口数量；
        for attr in self.w_m.keys():
            i ,j = list(attr)[0] , list(attr)[1]
            temp = self.w_m[i,j] / self.data.loc[i,'格口处理货量']
            self.constraint['c13_1',i+j] = self.m.addConstr(self.g_m[i,j] - temp >= -eps, name='c13_1_'+i+j)
            self.constraint['c13_2',i+j] = self.m.addConstr(self.g_m[i,j] - temp <= 1-eps, name='c13_2_'+i+j)
        ## 式（11.15）表示对于每个场地而言，所有需要机器处理的小件包裹量所需格口不超过当前场地设备现状隔口数量。
        temp = {}
        for attr in self.g_m.keys():
            i = list(attr)[0]
            j = list(attr)[1]
            if i in temp.keys():
                temp[i] += self.g_m[attr]
            else:
                temp[i] = self.g_m[attr]    
        for i in temp.keys():
            self.constraint['c14',i] = self.m.addConstr(self.data.loc[i,'分拣机格口数'] - temp[i] >= -eps, name='c14_'+i)

        # 设定目标函数
        ## 增加惩罚项
        temp = {}
        for (rount_path,pack_path) in self.x.keys():
            for index in range(len(pack_path)-1):
                begin = pack_path[index]
                end   = pack_path[index+1]
                if (begin,end) in temp.keys():
                    temp[begin,end] += self.x[rount_path,pack_path]
                else:
                    temp[begin,end] = self.x[rount_path,pack_path]
        penalty = gp.LinExpr()
        for attrs in temp.keys():
            penalty += temp[attrs]**2
        obj = gp.LinExpr()
        for i in self.z_a.keys():
            obj += self.z_a[i] * self.data.loc[i,'人工成本'] + self.z_m[i] * self.data.loc[i,'机器成本']
        self.m.setObjective(obj - penalty*self.M, GRB.MINIMIZE)


    def solve_model(self):
        self.m.Params.MIPGap = 0.01
        self.m.Params.TimeLimit = 60
        self.m.Params.Heuristics = 1
        self.m.Params.Method = 2
        self.m.Params.MIPFocus = 1
        self.m.Params.NonConvex = 2
        self.m.write('pack_model.lp')
        self.m.optimize()
        self.status = self.m.status
        if self.m.status not in {2, 7, 8, 9, 10, 13, 15}:
            print(f'模型求解失败,模型状态:{self.m.status}')
            self.m.computeIIS()
            self.m.write("model_file2.ilp")
        else:
            print(f'模型存在可行解,模型状态:{self.m.status}')
            # 存储解
            self.solution = self.get_solution(self.x)
            self.obj = self.m.objVal
            # 存储建包数量
            self.a_qty =  self.get_solution(self.z_a)
            self.m_qty =  self.get_solution(self.z_m)
            self.uu = self.get_solution(self.u)
            self.yy = self.get_solution(self.y)
            self.ya = self.get_solution(self.y_a)
            self.ym = self.get_solution(self.y_m)
            self.a_w = self.get_solution(self.w_a)
            self.m_w = self.get_solution(self.w_m)

            
            # 输出结果
            print('建包路径：')
            for line in self.solution.keys():
                if self.solution[line] == 1.0:
                    print(line)
            # print('集包分配：人工/机器')
            # for attr in self.a_qty.keys():
            #     print(attr,'     :',self.a_qty[attr],'/',self.m_qty[attr])
            print('目标值为：',self.obj)

    def get_solution(self,x:dict)->dict:
        y = {}
        for attr in x.keys():
            y[attr]  = int(x[attr].x)
        return y


    def _get_pack_path_set(self,point1,point2):
        paths = []
        for rout in self.packages_path.keys():
            for pack in self.packages_path[rout]:
                if point1 in pack and point2 in pack:
                    paths.append(pack)
        return paths

    def _get_rout_path_set(self,point1,point2):
        paths = []
        for rout in self.packages_path.keys():
            if point1 in rout and point2 in rout and point2+point1 not in rout:
                    paths.append(rout)
        return paths
    
    # def draw(self):
    #     self.data['机器流量'] = self.data.apply(lambda x:x['分拣机格口数']*x['格口处理货量'],axis=1 )
    #     self.data['流量'] = self.data.apply(lambda x:10000*x['人工流量']+x['机器流量'],axis=1 )
    #     # 绘图模块
    #     nodes = []
    #     for index,x in self.data.iterrows():
    #         temp = {"name":index,"symbolSize":x['流量']/600}
    #         nodes.append(temp)
    #     # 构建边数据
    #     links = []
    #     for line in self.rout_list:
    #         for index in range(len(line)-1):
    #             temp = {"source":line[index] , "target":line[index+1]}
    #             # if temp not in links:
    #             links.append(temp)
    #     # 创建关系网络图对象
    #     self.graph = Graph().add("", nodes, links, 
    #                             repulsion=10000,
    #                             # is_rotate_label=True,
    #                             linestyle_opts=opts.LineStyleOpts(color="source", curve=0.1),
    #                             label_opts=opts.LabelOpts(position="right")).set_global_opts(title_opts=opts.TitleOpts(title="关系网图"))
    #     return self.graph.render_notebook()