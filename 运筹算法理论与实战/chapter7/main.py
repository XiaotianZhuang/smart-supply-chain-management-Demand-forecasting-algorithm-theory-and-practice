from model import *
import seaborn as sns

if __name__ == '__main__':
    m = model('./数据.xlsx')
    m.set_variables()
    m.set_constraint()
    m.set_object()
    m.solve_model()
    if m.status == 2:
        print('-----------存在最优解-----------')
        print('路由选择：')
        print_sol(m.solution_y)
        print('线路选择：')
        print_sol(m.solution_z)
        print('车辆选择：')
        print_sol(m.solution_p)
        print('车辆数量：')
        print_sol(m.solution_q)
    else:
        print('求解失败')