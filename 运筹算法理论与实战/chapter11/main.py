from model import *

if __name__ == '__main__':
    # 读取数据
    data = model('./数据.xlsx',{'ACEF':8000,
                            'ACEG':12000,
                            'ACDH':10000,
                            'BDEF':9000,
                            'BDEG':5000,
                            'BDH':12000})
    #建立模型
    data.set_model()
    # 求解模型
    data.solve_model()
    print('x',data.solution)