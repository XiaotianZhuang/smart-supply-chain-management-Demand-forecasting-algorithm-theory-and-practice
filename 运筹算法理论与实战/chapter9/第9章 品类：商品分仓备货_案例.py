import pandas as pd
from collections import defaultdict
import random
import gurobipy as gp
from gurobipy import GRB

# Excel文件路径
file_path = '第9章 品类：商品分仓备货.xlsx'
# 使用ExcelFile读取Excel文件
xls = pd.ExcelFile(file_path)
# 获取所有sheet的名称
sheet_names = xls.sheet_names
# 使用一个字典来存储所有的DataFrame
sheets_data = {}
# 遍历所有的sheet名称，读取每个sheet到不同的DataFrame
for sheet_name in sheet_names:
    # 读取每个sheet
    df = pd.read_excel(xls, sheet_name=sheet_name)
    # 将读取的DataFrame存储到字典中，键为sheet的名称
    sheets_data[sheet_name] = df
# 关闭ExcelFile
xls.close()
cate = sheets_data["品类信息数据"]
store = sheets_data["仓库上限信息"]
corr = sheets_data["品类相关度"]


class InputData:
    def __init__(self, wh_constraints, total_item, corr):
        # 表相关，保存输入源表
        # 仓约束表
        self.wh_constraints = wh_constraints.copy()
        # 品类全集表
        self.total_item = total_item.copy()
        # 关联度表
        self.corr = corr.copy()

        # 提取出仓库与品类的集合
        self.cate_set = set(total_item["品类名称"])
        self.wh_set = set(wh_constraints["仓库名称"])

        # 时间集合
        self.T = {4, 5, 6}

        # 品类的相关
        self.inv_dict = defaultdict(int)
        self.vol_dict = defaultdict(float)
        self.sku_num_dict = defaultdict(int)
        self.porduce_dict = defaultdict(int)
        self.cate_type_proportion_dict = defaultdict(int)

        columns1 = ['品类名称', '4月库存', '5月库存', '6月库存', '4月产能', '5月产能',
                    '6月产能', '5月sku数量', '6月sku数量', '4月sku数量', '4月体积', '5月体积', '6月体积']
        total_item_cate = self.total_item[columns1].groupby('品类名称').sum()
        for i, row in total_item_cate.iterrows():
            self.inv_dict[4, i] = row['4月库存']
            self.inv_dict[5, i] = row['5月库存']
            self.inv_dict[6, i] = row['6月库存']
            self.vol_dict[4, i] = row['4月体积']
            self.vol_dict[5, i] = row['5月体积']
            self.vol_dict[6, i] = row['6月体积']
            self.sku_num_dict[4, i] = row['4月sku数量']
            self.sku_num_dict[5, i] = row['5月sku数量']
            self.sku_num_dict[6, i] = row['6月sku数量']
            self.porduce_dict[4, i] = row['4月产能']
            self.porduce_dict[5, i] = row['5月产能']
            self.porduce_dict[6, i] = row['6月产能']

        total_item_type = total_item[['品类名称', '4月库存', 'A件型占比', 'B件型占比', 'C件型占比']].copy()
        total_item_type["A件型库存"] = total_item_type['A件型占比'] * total_item_type['4月库存']
        total_item_type["B件型库存"] = total_item_type['B件型占比'] * total_item_type['4月库存']
        total_item_type["C件型库存"] = total_item_type['C件型占比'] * total_item_type['4月库存']
        total_item_type = total_item_type[['品类名称', '4月库存', "A件型库存", "B件型库存", "C件型库存"]].groupby(
            '品类名称').sum()
        for i, row in total_item_type.iterrows():
            self.cate_type_proportion_dict[i, "A"] = row["A件型库存"] / row['4月库存']
            self.cate_type_proportion_dict[i, "B"] = row["B件型库存"] / row['4月库存']
            self.cate_type_proportion_dict[i, "C"] = row["C件型库存"] / row['4月库存']

        # 仓库相关
        self.inv_upp_dict = defaultdict(int)
        self.vol_upp_dict = defaultdict(float)
        self.sku_num_upp_dict = defaultdict(int)
        self.porduce_upp_dict = defaultdict(int)
        for i, row in self.wh_constraints.iterrows():
            self.inv_upp_dict[row["仓库名称"]] = row["库存上限"]
            self.vol_upp_dict[row["仓库名称"]] = row["体积上限"]
            self.sku_num_upp_dict[row["仓库名称"]] = row["sku数量上限"]
            self.porduce_upp_dict[row["仓库名称"]] = row["产能上限"]

        # 最大分仓数，随机生成
        self.cate_max_store_num = defaultdict(int)
        random.seed = 1
        for i in self.cate_set:
            self.cate_max_store_num[i] = random.randint(1, 6)

        # 每个仓的件型占比
        self.wh_A_type = {"一号仓库": 0.6, "二号仓库": 0.6, "三号仓库": 0, "四号仓库": 0, "五号仓库": 0, "六号仓库": 0}
        self.wh_B_type = {"一号仓库": 0, "二号仓库": 0, "三号仓库": 0, "四号仓库": 0, "五号仓库": 0.5, "六号仓库": 0.5}

        # 主营品,大于80%'品类名称'
        self.main_wh_cate = defaultdict(set)
        for j in self.wh_set:
            for i, row in total_item.iterrows():
                if row["4月库存"] / total_item_type.loc[row["品类名称"]]["4月库存"] > 0.8 and row["仓库名称"] == j:
                    self.main_wh_cate[j].add(row["品类名称"])
        # 品类关联度
        self.dict_corr_cate_cate = defaultdict(int)
        for i, row in corr.iterrows():
            self.dict_corr_cate_cate[row["品类名称A"], row["品类名称B"]] = row["相关度"]
        # 主营仓库
        self.main_wh_set = set(["一号仓库", "二号仓库", "三号仓库", "四号仓库"])  # 主营仓库集合
        # 不能在一起的品类
        self.not_in_one_wh = [["饮用水", "组装电脑"], ["手机", "饮用水"]]

        # 不能放该仓的品类
        self.wh_not_in_cate = {"一号仓库": ["口罩", "耳罩/耳包"]}
        # 只能放该仓的品类
        self.cate_only_in_wh = {"二号仓库": ["插座", "线缆"]}


class ItemPlanBaseModelNew:
    @staticmethod
    def handle(input_data: InputData):
        CORR = 0
        m = gp.Model()
        final_solution_of_x = {}
        final_solution_of_y = defaultdict(int)

        # x3:三级分仓0-1变量

        x = m.addVars(input_data.cate_set, input_data.wh_set,
                      vtype=GRB.CONTINUOUS, name='x', lb=0, ub=1)
        y1 = m.addVars(input_data.cate_set, input_data.wh_set, vtype=GRB.BINARY, name="yij")

        y2 = {}
        # 品类1和品类2关联度
        for cate1 in input_data.cate_set:
            for cate2 in input_data.cate_set:
                if input_data.dict_corr_cate_cate[cate1, cate2] > CORR and cate1 != cate2:
                    y2[cate1, cate2] = m.addVar(vtype=GRB.BINARY, name=f'y{cate1}{cate2}')
        u = m.addVars(input_data.wh_set, vtype=GRB.BINARY, name="u")
        """
        模型约束
        """
        m.addConstrs((gp.quicksum(x[i, j] for j in input_data.wh_set) == 1
                      for i in input_data.cate_set), name="sum_cate")

        m.addConstrs((y1[i, j] >= x[i, j] for j in input_data.wh_set for i in input_data.cate_set), name="y与x关系")

        # 仓库上限
        for t in input_data.T:
            m.addConstrs((gp.quicksum(x[i, j] * input_data.inv_dict[t, i] for i in input_data.cate_set) <=
                          input_data.inv_upp_dict[j] for j in input_data.wh_set), name=f"库存上限{t}")
            m.addConstrs((gp.quicksum(x[i, j] * input_data.vol_dict[t, i] for i in input_data.cate_set) <=
                          input_data.vol_upp_dict[j] for j in input_data.wh_set), name=f"体积上限{t}")
            m.addConstrs((gp.quicksum(x[i, j] * input_data.sku_num_dict[t, i] for i in input_data.cate_set) <=
                          input_data.sku_num_upp_dict[j] for j in input_data.wh_set), name=f"sku数量上限{t}")
            m.addConstrs((gp.quicksum(x[i, j] * input_data.porduce_dict[t, i] for i in input_data.cate_set) <=
                          input_data.porduce_upp_dict[j] for j in input_data.wh_set), name=f"产能上限{t}")

        # 分仓数量限制
        m.addConstrs((gp.quicksum(y1[i, j] for j in input_data.wh_set) <= input_data.cate_max_store_num[i] for i in
                      input_data.cate_set), name="最大分仓数限制")

        #书本案例结果对应的件型限制如下：
        m.addConstrs(gp.quicksum(x[i, j] * input_data.cate_type_proportion_dict[i, "A"] * max(
            input_data.inv_dict[t, i] for t in input_data.T) for i in input_data.cate_set) >= input_data.wh_A_type[j] *
                     input_data.inv_upp_dict[j] * u[j]
                     for j in input_data.wh_set)  # A件型占比
        m.addConstrs(gp.quicksum(x[i, j] * input_data.cate_type_proportion_dict[i, "B"] * max(
            input_data.inv_dict[t, i] for t in input_data.T) for i in input_data.cate_set) >= input_data.wh_B_type[j] *
                     input_data.inv_upp_dict[j] * u[j]
                     for j in input_data.wh_set)  # B件型占比
         # # 如果与书中前面章节件型限制约束相同，可修改为对应的件型限制如下：
        # m.addConstrs(gp.quicksum(x[i, j] * input_data.cate_type_proportion_dict[i, "A"] *
        #     input_data.inv_dict[t, i] for i in input_data.cate_set) >= input_data.wh_A_type[j] *
        #              input_data.inv_upp_dict[j] * u[j]
        #              for j in input_data.wh_set for t in input_data.T)  # A件型占比
        # m.addConstrs(gp.quicksum(x[i, j] * input_data.cate_type_proportion_dict[i, "B"] *
        #     input_data.inv_dict[t, i]  for i in input_data.cate_set) >= input_data.wh_B_type[j] *
        #              input_data.inv_upp_dict[j] * u[j]
        #              for j in input_data.wh_set for t in input_data.T)  # B件型占比

        M = 100000000
        # 仓库是否使用约束
        m.addConstrs(
            gp.quicksum(y1[i, j] for i in input_data.cate_set) >= M * (u[j] - 1) + 1 for j in input_data.wh_set)
        m.addConstrs(gp.quicksum(y1[i, j] for i in input_data.cate_set) <= M * u[j] for j in input_data.wh_set)

        # 主营仓库
        m.addConstrs(
            u[j1] >= u[j2] for j1 in input_data.main_wh_set for j2 in (input_data.wh_set - input_data.main_wh_set))

        # 不能在一起的品类
        m.addConstrs(y1[i1, j] + y1[i2, j] <= 1 for i1, i2 in input_data.not_in_one_wh for j in input_data.wh_set)

        # 不能放该仓的品类
        m.addConstrs(y1[i, j] == 0 for j in input_data.wh_not_in_cate.keys() for i in input_data.wh_not_in_cate[j])
        # 只能放该仓的品类
        m.addConstrs(x[i, j] == 1 for j in input_data.cate_only_in_wh.keys() for i in input_data.cate_only_in_wh[j])

        # 设置目标函数
        S = 10
        cof = sum(input_data.dict_corr_cate_cate.values())
        m.setObjective(gp.quicksum(
            input_data.dict_corr_cate_cate[i1, i2] * y2[i1, i2] for i1 in input_data.cate_set for i2 in
            input_data.cate_set if
            (i1 != i2 and input_data.dict_corr_cate_cate[i1, i2] > CORR)) - 2 * S * gp.quicksum(
            u[j] for j in input_data.wh_set) - 2 * cof * gp.quicksum(
            y1[i, j] for j in input_data.wh_set for i in input_data.cate_set) + cof * gp.quicksum(
            y1[i, j] for j in input_data.main_wh_cate.keys() for i in input_data.main_wh_cate[j]), GRB.MAXIMIZE)

        # 模型参数设定
        # m.Params.MIPGap = 0.0002
        m.Params.TimeLimit = 60 * 10
        m.Params.Heuristics = 1
        m.Params.Method = 2
        m.Params.NodeMethod = 2
        m.Params.MIPFocus = 1
        m.write("model.lp")
        m.optimize()

        # 输出解
        num = 0
        for i in input_data.cate_set:
            for j in input_data.wh_set:
                if x[i, j].x > 0:
                    final_solution_of_x[num] = (i, j, x[i, j].x)
                    final_solution_of_y[num] = (i, j, y1[i, j].x)
                    num += 1

        corr_before = 0#优化前关联度
        for j in set(cate["仓库名称"]):
            cate_set = set(cate[cate["仓库名称"] == j]["品类名称"])
            for i1 in cate_set:
                for i2 in cate_set:
                    corr_before += input_data.dict_corr_cate_cate[i1, i2]

        corr_after = 0  # 优化后的关联度
        for i1 in input_data.cate_set:
            for i2 in input_data.cate_set:
                if i1 != i2 and input_data.dict_corr_cate_cate[i1, i2] > CORR:
                    if y2[i1, i2].x > 0 :
                        corr_after += input_data.dict_corr_cate_cate[i1, i2] * y2[i1, i2].x

        return final_solution_of_x, final_solution_of_y, corr_before, corr_after


input_data1 = InputData(store, cate, corr)

model1 = ItemPlanBaseModelNew()
final_solution_of_x, final_solution_of_y, corr_before, corr_after = model1.handle(input_data1)
# 加工输入输出
df1 = pd.DataFrame(final_solution_of_x)
df2 = pd.DataFrame(final_solution_of_y)
df1 = df1.T
df2 = df2.T
df1 = df1.rename(columns={0: "品类名称", 1: "仓库名称", 2: "品类比例"})
df2 = df2.rename(columns={0: "品类名称", 1: "仓库名称", 2: "是否在仓库"})
column1 = ['品类名称', '4月库存', '4月产能', '4月sku数量', '4月体积', '5月库存', '5月产能', '5月sku数量', '5月体积', '6月库存', '6月产能', '6月sku数量', '6月体积']

cate_all = cate[column1].groupby(['品类名称']).sum().reset_index()
result_x = df1.merge(cate_all, on="品类名称", how="left")
# 匹配上原始数据
with pd.ExcelWriter('商品分仓结果.xlsx') as writer:
    result_x.to_excel(writer, sheet_name='品类分仓详情', index=False)
    df2.to_excel(writer, sheet_name='品类仓库中出现', index=False)

print("优化前的关联度: ", corr_before)
print("优化后的关联度：", corr_after)