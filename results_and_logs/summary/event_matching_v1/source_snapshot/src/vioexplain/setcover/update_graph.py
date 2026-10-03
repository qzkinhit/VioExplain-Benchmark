# coding=utf-8
"""基于有向图 + DFS 的知识更新算法 Update。

重构自原型 legacy/setcovering/Update.py，算法逐行保持，仅：
 1. 复用 models.Constraint/Reason、apriori.*；
 2. Update 返回 (F_f, update_num)，供实验编排读取更新结果；
 3. 演示数据移入 __main__。

思路（博士论文 §5.5 Algo5-3/5-4）：以违反特征为顶点、关联规则为有向边建图，
DFS 把未覆盖约束 D 按关联并入已有原因（扩充知识 Expanded Knowledge）或生成新原因（全新知识 New Knowledge），
并按更新次数初始化可能性权重 w=1/update_num。
"""
from .models import Constraint, Reason
from .apriori import Generate_L, Generate_big_rules


class Vertex:
    def __init__(self, key):
        self.id = key
        self.connectedTo = {}

    def addNeighbor(self, nbr, weight=0):
        self.connectedTo[nbr] = weight

    def getConnection(self):
        return self.connectedTo.keys()

    def getId(self):
        return self.id

    def getWeight(self, nbr):
        return self.connectedTo[nbr]

    def setColor(self, color):
        self.color = color

    def getColor(self):
        return self.color


class Graph:
    def __init__(self):
        self.vertList = {}
        self.numVertices = 0

    def addVertex(self, key):
        self.numVertices += 1
        newVertex = Vertex(key)
        self.vertList[key] = newVertex
        return newVertex

    def getVertex(self, n):
        return self.vertList.get(n)

    def __contains__(self, n):
        return n in self.vertList

    def addEdge(self, f, t, cost=0):
        if f not in self.vertList:
            self.addVertex(f)
        if t not in self.vertList:
            self.addVertex(t)
        self.vertList[f].addNeighbor(self.vertList[t], cost)

    def getVertices(self):
        return self.vertList.values()

    def __iter__(self):
        return iter(self.vertList.values())


def Item_in_F(f, item):
    count = 0
    for c in item:
        for cons in f.f_cover:
            if c.name == cons.name:
                count += 1
    return len(item) == count


def Calcu_L(c, F_f):
    """计算 L(c)：覆盖约束 c 的所有原因集合。"""
    retSet = set()
    for f in F_f:
        for cons in f.f_cover:
            if cons.name == c.name:
                retSet.add(f)
                break
    return retSet


def Is_in_f(f, c):
    for cons in f.f_cover:
        if cons.name == c.name:
            return True
    return False


def Update(R_cons, F_f, rule_list, B_cons, D_cons, C_cons, min_conf):
    """知识更新主算法。返回 (F_f, update_num)。"""
    update_num = {}
    g = Graph()
    for c in R_cons:
        g.addVertex(c)
    for item in rule_list:
        if len(item[0]) == 1:
            for con in item[1]:
                g.addEdge(con, list(item[0])[0], item[2])
        elif len(item[0]) > 1:
            for f in F_f:
                if Item_in_F(f, item[0]):
                    for con in item[1]:
                        g.addEdge(con, list(item[0])[0], item[2])

    waitlist = []
    D = set()
    R = set()
    B = set()
    C = set()
    for con in C_cons:
        for ver in g.getVertices():
            if ver.id.name == con.name:
                C.add(ver)
    for cons in D_cons:
        for ver in g.getVertices():
            if ver.id.name == cons.name:
                D.add(ver)
    D = D.copy()
    for ver in g.getVertices():
        R.add(ver)
    R = R.copy()
    D = set(D) | (set(R) & set(B))

    for aVertex in g:
        aVertex.setColor('white')
    for aVertex in D:
        if aVertex.getColor() == 'white':
            aVertex.setColor("black")
            update_num = dfsvisit(aVertex, D, C, B, waitlist, F_f, min_conf, g, update_num)

    # 初始化可能性权重 w
    for key in update_num.keys():
        for f in F_f:
            for cons in f.f_cover:
                if key == cons:
                    cons.w = 1 / update_num[key]
    return F_f, update_num


def dfsvisit(startVertex, D, C, B, waitlist, F_f, min_conf, g, update_num):
    if startVertex.getColor() == 'white':
        startVertex.setColor('gray')
    if len(startVertex.getConnection()) == 0 and startVertex in D:
        f = Reason("unnamed_new_f", {startVertex.id})
        B = B | {startVertex}
        F_f.append(f)
    for nextVertex in startVertex.getConnection():
        if nextVertex.getColor() != 'black':
            if nextVertex in C and startVertex in D:
                for f in F_f:
                    if Is_in_f(f, nextVertex.id):
                        temp_cons = Constraint(startVertex.id.name, startVertex.id.feature1, startVertex.id.feature2)
                        f.f_cover.add(temp_cons)
                        update_num[temp_cons] = update_num.get(temp_cons, 0) + 1
                    for x in waitlist:
                        if x[0].id not in f.f_cover:
                            if x[1] * g.getVertex(startVertex.id).getWeight(nextVertex) > min_conf:
                                temp_cons = Constraint(x[0].id.name, x[0].id.feature1, x[0].id.feature2)
                                f.f_cover.add(temp_cons)
                                update_num[temp_cons] = update_num.get(temp_cons, 0) + 1
                                B = B | {x[0]}
                B = B | {startVertex}

            elif nextVertex in D:
                if len(nextVertex.getConnection()) == 0:
                    temp_cons = Constraint(nextVertex.id.name, nextVertex.id.feature1, nextVertex.id.feature2)
                    f = Reason("unnamed_new_f", {temp_cons})
                    D = D - {nextVertex.id}
                    B = B | {nextVertex.id}
                    for x in waitlist:
                        if x[1] * g.getVertex(startVertex.id).getWeight(nextVertex) > min_conf:
                            f.f_cover.add(Constraint(x[0].id.name, x[0].id.feature1, x[0].id.feature2, 1))
                            update_num[temp_cons] = update_num.get(temp_cons, 0) + 1
                            B = B | {x[0]}
                    F_f.append(f)
                else:
                    waitlist.append((startVertex, g.getVertex(startVertex.id).getWeight(nextVertex)))

            elif nextVertex in C and startVertex in C:
                A = set()
                for x in waitlist:
                    if x[1] * g.getVertex(startVertex.id).getWeight(nextVertex) > min_conf:
                        A.add(x[0])
                        B = B | {x[0]}
                L_start = Calcu_L(startVertex.id, F_f)
                L_next = Calcu_L(nextVertex.id, F_f)
                L_sub = L_next - L_start
                for c in A:
                    for f in L_sub:
                        flag = 0
                        for cons in f.f_cover:
                            if c.id.name == cons.name:
                                flag = 1
                                break
                        if flag == 0:
                            temp_cons = Constraint(c.id.name, c.id.feature1, c.id.feature2, 1)
                            f.f_cover.add(temp_cons)
                            update_num[temp_cons] = update_num.get(temp_cons, 0) + 1
                        B = B | {c}

            dfsvisit(nextVertex, D, C, B, waitlist.copy(), F_f, min_conf, g, update_num)
    return update_num


if __name__ == "__main__":
    # 演示数据（自原型 Update.py 末尾搬入）
    c1 = Constraint("a", 0, 2.5, 1); c2 = Constraint("b", -1, 2, 1); c3 = Constraint("c", -3, 0, 1)
    c4 = Constraint("d", -2, 4, 1); c5 = Constraint("e", -4, 5, 1); c6 = Constraint("f", 0, 3, 1)
    c7 = Constraint("g", -2.7, 0, 1); c8 = Constraint("h", -2, 2, 1); c9 = Constraint("i", -7, 3, 1)
    c10 = Constraint("j", 0, 10, 1); c11 = Constraint("k", -1, 5, 1)
    c1f = Constraint("a", 0, 2); c1ff = Constraint("a", 0, 2.5); c2f = Constraint("b", -1.2, 2.1)
    c3f = Constraint("c", -2.5, 0); c4f = Constraint("d", -2.3, 3.5); c4ff = Constraint("d", -2.1, 3.5)
    c4fff = Constraint("d", -2, 3.7); c5f = Constraint("e", -3.5, 6); c6f = Constraint("f", 0, 3.2)
    c6ff = Constraint("f", -1, 3); c7ff = Constraint("g", -2.8, 0.2); c8f = Constraint("h", -2.5, 2)
    c9f = Constraint("i", -6, 3.2); c10f = Constraint("j", -1, 11); c11f = Constraint("k", -0.4, 6)
    F_f = [Reason('A', {c1f, c3f}), Reason('B', {c2f, c4f, c8f}), Reason('C', {c3f, c6f, c7ff}),
           Reason('D', {c7ff, c4ff}), Reason('E', {c6ff, c9f}), Reason('F', {c1ff, c10f}),
           Reason('G', {c4fff, c11f})]
    constraint = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11]
    data_set = [[c1, c2, c5], [c2, c3, c5], [c2, c4, c5], [c6, c7, c9], [c6, c8, c9],
                [c9, c10, c11], [c1, c2, c4, c5], [c1, c2, c5, c11], [c1, c5, c6], [c9, c11]]
    L, support_data = Generate_L(data_set, k=3, min_support=0.2)
    big_rules_list = Generate_big_rules(L, support_data, min_conf=0.7)
    F_f, _ = Update(constraint, F_f, big_rules_list, set(), {c5}, {c1, c2, c3, c4, c6, c7, c8, c9, c10, c11}, 0.6)
    for f in F_f:
        print(f.name, ':', ' '.join(cons.name for cons in f.f_cover))
