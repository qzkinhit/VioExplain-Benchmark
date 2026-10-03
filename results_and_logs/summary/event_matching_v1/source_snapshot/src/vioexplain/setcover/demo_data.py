# coding=utf-8
"""6 组演示数据（逐字搬自原型 legacy/setcovering/SetCovering.py 末尾）。

用途：验证重构版 set_covering.MyCover 的输出与原型完全一致（解集代价 + 原因名）。
以函数形式返回 fresh 对象，避免跨调用被 MyCover 原地修改。
"""
from .models import Constraint, Reason
from .apriori import Generate_L, Generate_big_rules


def demo_constraints():
    c1 = Constraint("a", 0, 2.5, 1); c2 = Constraint("b", -1, 2, 1)
    c3 = Constraint("c", -3, 0, 1); c4 = Constraint("d", -2, 4, 1)
    c5 = Constraint("e", -4, 5, 1); c6 = Constraint("f", 0, 3, 1)
    c7 = Constraint("g", -2.7, 0, 1); c8 = Constraint("h", -2, 2, 1)
    c9 = Constraint("i", -7, 3, 1); c10 = Constraint("j", 0, 10, 1)
    c11 = Constraint("k", -1, 5, 1)
    return [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11]


def demo_rules():
    c = demo_constraints()
    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11 = c
    data_set = [[c1, c2, c5], [c2, c3, c5], [c2, c4, c5], [c6, c7, c9], [c6, c8, c9],
                [c9, c10, c11], [c1, c2, c4, c5], [c1, c2, c5, c11], [c1, c5, c6], [c9, c11]]
    L, support_data = Generate_L(data_set, k=3, min_support=0.2)
    return Generate_big_rules(L, support_data, min_conf=0.7)


def _F_f1():
    c1f = Constraint("a", 0, 2); c1ff = Constraint("a", 0, 2.5)
    c2f = Constraint("b", -1.2, 2.1); c3f = Constraint("c", -2.5, 0)
    c4f = Constraint("d", -2.3, 3.5); c4ff = Constraint("d", -2.1, 3.5); c4fff = Constraint("d", -2, 3.7)
    c5f = Constraint("e", -3.5, 6); c5ff = Constraint("e", -3.7, 4.5)
    c6f = Constraint("f", 0, 3.2); c6ff = Constraint("f", -1, 3)
    c7f = Constraint("g", -3, 0); c7ff = Constraint("g", -2.8, 0.2)
    c8f = Constraint("h", -2.5, 2); c9f = Constraint("i", -6, 3.2)
    c10f = Constraint("j", -1, 11); c11f = Constraint("k", -0.4, 6)
    return [Reason('A', {c1f, c3f}), Reason('B', {c2f, c4f, c5f, c8f}),
            Reason('C', {c3f, c6f, c7f}), Reason('D', {c7ff, c4ff}),
            Reason('E', {c6ff, c9f}), Reason('F', {c1ff, c10f}),
            Reason('G', {c4fff, c5ff, c11f})]


def _F_f2():
    c1f2 = Constraint("a", 0, 2); c1ff2 = Constraint("a", 0, 2.1)
    c2f2 = Constraint("b", -1.2, 2.1); c2ff2 = Constraint("b", -1.3, 2)
    c3f2 = Constraint("c", -2.5, 0.1); c4f2 = Constraint("d", -2.4, 3.6); c4ff2 = Constraint("d", -2.2, 3.8)
    c5f2 = Constraint("e", -3.4, 4.3); c5ff2 = Constraint("e", -3.2, 5.3)
    c6f2 = Constraint("f", 0, 3.2); c6ff2 = Constraint("f", 0, 3.4)
    c7f2 = Constraint("g", -3, 0.1); c7ff2 = Constraint("g", -2.5, 0)
    c8f2 = Constraint("h", -2.4, 2.1); c9f2 = Constraint("i", -5.9, 3.2); c9ff2 = Constraint("i", -5.3, 3)
    c10f2 = Constraint("j", -1, 12); c11f2 = Constraint("k", -0.4, 6)
    c11ff2 = Constraint("k", -0.3, 5.8); c11fff2 = Constraint("k", -0.2, 4.2); c11ffff2 = Constraint("k", -1.2, 5.3)
    return [Reason('A', {c1f2, c4f2, c6f2, c11f2}), Reason('B', {c1ff2, c4ff2}),
            Reason('C', {c6ff2, c11ff2}), Reason('D', {c2f2, c3f2, c5f2}),
            Reason('E', {c7f2, c8f2, c9f2}), Reason('F', {c2ff2, c10f2, c11fff2}),
            Reason('G', {c5ff2, c9ff2}), Reason('H', {c7ff2, c11ffff2})]


def _F_f3():
    c1f3 = Constraint("a", 0, 2); c2f3 = Constraint('d', -2.4, 3.6); c3f3 = Constraint('f', 0, 3.2)
    c4f3 = Constraint('k', -0.4, 6); c5f3 = Constraint('a', 0, 2.1); c6f3 = Constraint('d', -2.2, 3.8)
    c7f3 = Constraint('f', 0, 3.4); c9f3 = Constraint('k', -0.3, 5.8); c10f3 = Constraint('b', -1.2, 2.1)
    c11f3 = Constraint('c', -2.3, 0.1); c12f3 = Constraint('e', -3.5, 6.1); c13f3 = Constraint('h', -2.4, 2.1)
    c14f3 = Constraint('j', -1, 10.6); c15f3 = Constraint('b', -1.1, 2); c16f3 = Constraint('c', -2, 0.1)
    c17f3 = Constraint('e', -3.2, 5.9); c18f3 = Constraint('h', -2.3, 2); c19f3 = Constraint('j', 0, 11)
    c20f3 = Constraint('g', -2.4, 0.1); c21f3 = Constraint('i', -5.2, 3.2); c22f3 = Constraint('g', -2.1, 0)
    c23f3 = Constraint('i', -4.9, 3.1); c24f3 = Constraint('k', -0.3, 5.5)
    return [Reason('A', {c1f3, c2f3, c3f3, c4f3}), Reason('B', {c5f3, c6f3}),
            Reason('C', {c7f3, c9f3}), Reason('D', {c13f3, c10f3, c11f3, c12f3, c14f3}),
            Reason('E', {c15f3, c16f3}), Reason('F', {c18f3, c19f3, c17f3}),
            Reason('G', {c21f3, c20f3}), Reason('H', {c22f3, c23f3, c24f3})]


def _F_f4():
    d = {}
    spec = [('a',0,2),('c',-2.5,0),('b',-1.2,2.1),('d',-2.3,3.5),('e',-3.5,6),('h',-2.5,2),
            ('c',-2.5,0),('f',0,3.2),('g',-3,0),('g',-2.8,0.2),('d',-2.1,3.5),('f',-1,3),
            ('i',-6,3.2),('a',0,2.5),('j',-1,11),('d',-2,3.7),('e',-3.7,4.5),('k',-0.4,6),
            ('a',0,2),('c',-2.5,0),('d',-2.1,3.5),('b',-1.2,2.1),('e',-3.5,6),('h',-2.5,2),
            ('f',-1,3),('i',-6,3.2),('k',-0.4,6),('g',-3,0),('j',-1,11),('a',0,2),('d',-2.4,3.6),
            ('f',0,3.2),('k',-0.4,6),('a',0,2.1),('d',-2.2,3.8),('f',0,3.4),('k',-0.3,5.8),
            ('b',-1.2,2.1),('c',-2.5,0.1),('e',-3.4,4.3),('g',-3,0.1),('h',-2.4,2.1),('i',-5.9,3.2),
            ('b',-1.3,2),('j',-1,12),('k',-0.2,4.2),('e',-3.2,5.3),('i',-5.3,3),('g',-2.5,0),
            ('k',-1.2,5.3),('a',0,2),('d',-2.4,3.6),('f',0,3.2),('k',-0.4,6),('a',0,2.1),
            ('d',-2.2,3.8),('f',0,3.4),('k',-0.3,5.8),('b',-1.2,2.1),('c',-2.3,0.1),('e',-3.5,6.1),
            ('h',-2.4,2.1),('j',-1,10.6),('b',-1.1,2),('c',-2,0.1),('e',-3.2,5.9),('h',-2.3,2),
            ('j',0,11),('g',-2.4,0.1),('i',-5.2,3.2),('g',-2.1,0),('i',-4.9,3.1),('k',-0.3,5.5)]
    for i, (n, a, b) in enumerate(spec, 1):
        d[i] = Constraint(n, a, b)
    g = lambda *ids: {d[i] for i in ids}
    return [
        Reason('A', g(1,2)), Reason('B', g(3,4,5,6)), Reason('C', g(7,8,9)), Reason('D', g(10,11)),
        Reason('E', g(12,13)), Reason('F', g(14,15)), Reason('G', g(16,17,18)), Reason('H', g(19,20,21)),
        Reason('I', g(22,23,24)), Reason('J', g(25,26,27)), Reason('K', g(28,29)), Reason('L', g(30,31,32,33)),
        Reason('M', g(34,35)), Reason('N', g(36,37)), Reason('O', g(38,39,40)), Reason('P', g(41,42,43)),
        Reason('Q', g(44,45,46)), Reason('R', g(48,47)), Reason('S', g(50,49)), Reason('T', g(51,52,53,54)),
        Reason('U', g(56,55)), Reason('V', g(58,57)), Reason('W', g(59,60,61,62,63)), Reason('X', g(64,65)),
        Reason('Y', g(66,67,68)), Reason('Z', g(69,70)), Reason('A*', g(71,72,73)),
    ]


def _F_f5():
    spec = [('a',0,2),('c',-2.5,0),('b',-1.2,2.1),('d',-2.3,3.5),('e',-3.5,6),('h',-2.5,2),('c',-2.5,0),
            ('f',0,3.2),('g',-3,0),('g',-2.8,0.2),('d',-2.1,3.5),('f',-1,3),('i',-6,3.2),('a',0,2.5),
            ('j',-1,11),('d',-2,3.7),('e',-3.7,4.5),('k',-0.4,6),('a',0,2),('c',-2.5,0),('d',-2.1,3.5),
            ('b',-1.2,2.1),('e',-3.5,6),('h',-2.5,2),('f',-1,3),('i',-6,3.2),('k',-0.4,6),('g',-3,0),
            ('j',-1,11),('a',0,2),('d',-2.4,3.6),('f',0,3.2),('k',-0.4,6),('a',0,2.1),('d',-2.2,3.8),
            ('f',0,3.4),('k',-0.3,5.8),('b',-1.2,2.1),('c',-2.5,0.1),('e',-3.4,4.3),('g',-3,0.1),
            ('h',-2.4,2.1),('i',-5.9,3.2),('b',-1.3,2),('j',-1,12),('k',-0.2,4.2),('e',-3.2,5.3),
            ('i',-5.3,3),('g',-2.5,0),('k',-1.2,5.3),('a',0,2),('d',-2.4,3.6),('f',0,3.2),('k',-0.4,6),
            ('a',0,2.1),('d',-2.2,3.8),('f',0,3.4),('k',-0.3,5.8),('b',-1.2,2.1),('c',-2.3,0.1),
            ('e',-3.5,6.1),('h',-2.4,2.1),('j',-1,10.6),('b',-1.1,2),('c',-2,0.1),('e',-3.2,5.9),
            ('h',-2.3,2),('j',0,11),('g',-2.4,0.1),('i',-5.2,3.2),('g',-2.1,0),('i',-4.9,3.1),
            ('k',-0.3,5.5),('a',0.1,2.3),('k',-1.2,5.1),('j',-7.1,3),('b',-0.7,2.3),('c',-2.3,1),
            ('i',-6.5,2.3),('d',-2,4.5),('e',-3.1,4.3),('f',0,3),('g',-2.7,0.1),('h',-2.4,2.3),
            ('k',-0.5,5),('j',0,10),('b',-1,2),('c',-3,0),('f',0,3),('a',0,2.6),('j',0,10),('g',-2.6,0.1)]
    d = {i: Constraint(n, a, b) for i, (n, a, b) in enumerate(spec, 1)}
    g = lambda *ids: {d[i] for i in ids}
    return [
        Reason("A", g(1,2)), Reason("B", g(3,4,5,6)), Reason("C", g(7,8,9)), Reason("D", g(10,11)),
        Reason("E", g(12,13)), Reason("F", g(14,15)), Reason("G", g(16,17,18)), Reason("H", g(19,20,21)),
        Reason("I", g(22,23,24)), Reason("J", g(25,26,27)), Reason("K", g(28,29)), Reason("L", g(30,31,32,33)),
        Reason("M", g(34,35)), Reason("N", g(36,37)), Reason("O", g(38,39,40)), Reason("P", g(41,42,43)),
        Reason("Q", g(44,45,46)), Reason("R", g(47,48)), Reason("S", g(49,50)), Reason("T", g(51,52,53,54)),
        Reason("U", g(55,56)), Reason("V", g(57,58)), Reason("W", g(59,60,61,62,63)), Reason("X", g(64,65)),
        Reason("Y", g(66,67,68)), Reason("Z", g(69,70)), Reason("A*", g(71,72,73)), Reason("B*", g(74,75,76)),
        Reason("C*", g(77,78,79)), Reason("D*", g(80,81,82,83)), Reason("F*", g(84,85)),
        Reason("G*", g(86,87,88,89)), Reason("H*", g(90,91,92)),
    ]


def _F_f6():
    spec = [('a',0.1,2.3),('d',-1.3,3.8),('f',-0.3,3.2),('b',-1.2,2.3),('a',0,2.7),('e',-3.2,4.9),
            ('c',-2.4,0.2),('i',-6.4,2.7),('b',-0.5,2.4),('e',-4.3,4.9),('c',-3,0),('d',-2,4),
            ('e',-3.8,4.8),('j',-0.4,10.4),('k',-0.7,4.9),('j',0,11),('f',0.1,2.9),('k',-1,5),
            ('i',-6.7,3),('h',-2.5,2),('a',0,2),('c',-2,0),('g',-2.7,0),('b',-0.4,2),('g',-2.6,0.1),
            ('i',-6.5,2.5),('h',-2.4,2)]
    d = {i: Constraint(n, a, b) for i, (n, a, b) in enumerate(spec, 1)}
    g = lambda *ids: {d[i] for i in ids}
    return [Reason("A", g(1,2,3)), Reason("B", g(4,5,6)), Reason("C", g(7,8,9,10)), Reason("D", g(11,12)),
            Reason("E", g(13,14,15)), Reason("F", g(16,17)), Reason("G", g(18,19)), Reason("H", g(20,21,22)),
            Reason("I", g(23,24)), Reason("J", g(25,26,27))]


def demo_explanation_sets():
    return [_F_f1(), _F_f2(), _F_f3(), _F_f4(), _F_f5(), _F_f6()]
