# coding=utf-8
"""知识更新仿真的数据生成（重构自 legacy/ziju_0822/yset/generate.py，逐行保语义）。

generate_cause: 生成约束 + 异常原因，并按 drop_rate(=知识不完整度 inr%) 删除部分原因的表征，
                得到 org_cause(完整知识) 与 removed_cause(不完整知识)。
insert_error / calc_cost / update_cause_constraint / update_cause_y: 注错、求代价覆盖、知识更新。
"""
import copy
import random


class Constraint:
    def __init__(self, cons_id):
        self.id = cons_id
        self.violation = 0
        c01 = 0.3
        self.c = 0 if random.random() < c01 else (0, 0)
        multi_k = 0.5
        self.k = random.randint(2, 5) if random.random() < multi_k else 1

    def __str__(self):
        return "id={}, k={}, c={}, vio={}".format(self.id, self.k, self.c, self.violation)


class Cause:
    @staticmethod
    def _random_set_c(cons_type):
        if cons_type == tuple:
            rule = random.randint(0, 4)
            if rule == 0:
                return random.randint(-8, -1), 0
            elif rule == 1:
                return 0, random.randint(1, 8)
            elif rule == 2:
                return random.randint(-8, -1), random.randint(1, 8)
            elif rule == 3:
                x = random.randint(-7, -1); y = random.randint(-8, -1)
                x = x - 1 if x == y else x
                return min(x, y), max(x, y)
            else:
                x = random.randint(1, 8); y = random.randint(1, 7)
                y = y + 1 if x == y else y
                return min(x, y), max(x, y)
        else:
            return 1

    @staticmethod
    def _random_set_w():
        p_x = 0.7   # 0.3 概率为可能表征(可能性[0.4,0.8])，0.7 概率为确定表征(可能性1)，统一放 X
        return random.random() * 0.4 + 0.4 if random.random() > p_x else 1

    def __init__(self, cause_id, max_size, constraint_id_list, constraint_dict):
        x = random.sample(constraint_id_list, random.randint(2, max_size))
        self.cause_id = cause_id
        self.X = [(cid, Cause._random_set_c(type(constraint_dict[cid].c)), Cause._random_set_w())
                  for cid in x]
        self.Y = []                 # 新增表征（与 compare.py 的 Y 定义不同）
        self.answer_time = 0
        self.constraint_time = {}
        self.constraint_c = {}


def one_dis(c1, c2):
    if type(c1) == type(c2) == tuple:
        if c1[0] > c2[0]:
            c1, c2 = c2, c1
        if c2[1] <= c1[1]:
            return 1 - (c2[1] - c2[0]) / (c1[1] - c1[0])
        elif c2[0] <= c1[1]:
            return 1 - (c1[1] - c2[0]) / (c2[1] - c1[0])
        else:
            return 1
    else:
        return abs(c1 - c2)


def generate_cause(constraint_num, cause_num, drop_rate):
    """生成约束/原因，并删 drop_rate 比例原因的部分表征 → (constraint, org_cause, removed_cause)。"""
    constraint_id_list = list(range(constraint_num))
    constraint_dict = {cid: Constraint(cid) for cid in constraint_id_list}

    cause_id_list = list(range(cause_num))
    cause_dict = {cid: Cause(cid, int(max(constraint_num * 0.05, 4)),
                             constraint_id_list, constraint_dict) for cid in cause_id_list}

    removed_dict = copy.deepcopy(cause_dict)
    removed_list = random.sample(cause_id_list, int(len(cause_id_list) * drop_rate))
    for i in removed_list:
        total_cons_num = len(cause_dict[i].X)
        remove_num = int((total_cons_num + 3) / 4)   # 2-4:1, 5-8:2, 9-12:3 ...
        remove_cons = random.sample(removed_dict[i].X, remove_num)
        for cons in remove_cons:
            removed_dict[i].X.remove(cons)
    return constraint_dict, cause_dict, removed_dict


def insert_error(constraint_dict, cause_dict, select_num):
    def generate_c(one_c):
        if type(one_c) == tuple:
            x, y = 0, 0
            while x == 0 and y == 0:
                x, y = one_c
                x_range = min(abs(x) * 0.2, 1); y_range = min(abs(y) * 0.2, 1)
                x = x + (random.random() - 0.5) * x_range
                y = y + (random.random() - 0.5) * y_range
                x = 0 if abs(x) < 1 else x
                y = 0 if abs(y) < 1 else y
            return min(x, y), max(x, y)
        else:
            return 1

    for i in constraint_dict:
        constraint_dict[i].c = (0, 0) if type(constraint_dict[i].c) == tuple else 0
        constraint_dict[i].violation = 0

    select_cause_list = random.sample(list(cause_dict.keys()), select_num)
    violation_constraint = set()
    for select_cause in select_cause_list:
        for cid, c, w in cause_dict[select_cause].X:
            if random.random() < w:
                constraint_dict[cid].c = generate_c(c)
                constraint_dict[cid].violation = 1
                violation_constraint.add(cid)
    return select_cause_list, violation_constraint


def calc_cost(constraint, cause, multi_k=0, use_y=0):
    def cost_by_cause(cause_id):
        cost = 0
        for cid, c, w in cause[cause_id].X:
            k = 1 if multi_k == 0 else constraint[cid].k
            cost += w * k * one_dis(c, constraint[cid].c)
        if use_y:
            for cid, c, w in cause[cause_id].Y:
                k = 1 if multi_k == 0 else constraint[cid].k
                cost += w * k * one_dis(c, constraint[cid].c)
        return cost

    def cover_by_cause(cause_id):
        cover = set()
        for cid, c, w in cause[cause_id].X:
            if constraint[cid].violation and one_dis(c, constraint[cid].c) < 1:
                cover.add(cid)
        if use_y:
            for cid, c, w in cause[cause_id].Y:
                if constraint[cid].violation and one_dis(c, constraint[cid].c) < 1:
                    cover.add(cid)
        return cover

    cost_dict, cover_dict, total_cover = {}, {}, set()
    for cause_id in cause:
        cost_dict[cause_id] = cost_by_cause(cause_id)
        cover_dict[cause_id] = cover_by_cause(cause_id)
        total_cover |= cover_dict[cause_id]
    return cost_dict, cover_dict, total_cover


def update_cause_constraint(constraint, cause, cause_cover, cause_answer, uncovered_constraint):
    for cause_id in cause_answer:
        cause[cause_id].answer_time += 1
    for cid in uncovered_constraint:
        for cause_id in cause_answer:
            if cid not in cause_cover[cause_id]:
                if cid in cause[cause_id].constraint_time:
                    c_t = cause[cause_id].constraint_time[cid]
                    one_c = constraint[cid].c
                    old_c = cause[cause_id].constraint_c[cid]
                    if type(old_c) == tuple:
                        cause[cause_id].constraint_c[cid] = ((old_c[0] * c_t + one_c[0]) / (c_t + 1),
                                                             (old_c[1] * c_t + one_c[1]) / (c_t + 1))
                    else:
                        cause[cause_id].constraint_c[cid] = old_c
                    cause[cause_id].constraint_time[cid] += 1
                else:
                    cause[cause_id].constraint_time[cid] = 1
                    cause[cause_id].constraint_c[cid] = constraint[cid].c


def update_cause_y(cause, theta=0.3):
    for cause_id in cause:
        cause[cause_id].Y.clear()
        for cid in cause[cause_id].constraint_time:
            if cause[cause_id].answer_time > 5 and \
                    cause[cause_id].constraint_time[cid] / cause[cause_id].answer_time >= theta:
                cause[cause_id].Y.append((cid, cause[cause_id].constraint_c[cid],
                                          cause[cause_id].constraint_time[cid] / cause[cause_id].answer_time))
