from model import *


if __name__ == '__main__':
    m = model('./数据.xlsx')
    m.set_variables()
    m.set_object()
    m.set_constraint()
    m.solve_model()
    print(m.solution)
    m.solution.to_excel('sol2.xlsx')