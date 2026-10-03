# coding=utf-8
"""基于集合覆盖模型的违反解释算法 AEC（Anomaly Explanation by set Covering）。

重构自原型 legacy/setcovering/SetCovering.py，算法逐行保持，仅做工程化改造：
 1. MyCover 返回 (sumcost, H) 而非仅 print，供实验编排调用（verbose 可保留原型打印）；
 2. α(alpha)/β(beta)/γ(gamma)/THRESHOLD 由参数传入（默认值 = 原型 0.2/4/0.1/2），不再硬编码；
 3. 复用 models.Constraint/Reason、apriori.*（删去重复定义）；
 4. 6 组演示数据移入 __main__。

优先级（与博士论文 §5.4.2 / 方法改进文档一致）：
   先覆盖多序列约束(k>1, 按 k 降序) → 剩余按最小代价升序覆盖；
   Select 中权衡 最大覆盖 > 最小代价 > 关联关系(Relation)，并用 HasCovered⟨K,V⟩ 缓存复用相似原因。
"""
from . import models
from .apriori import Generate_L, Generate_big_rules, Set_Sub, Set_Inter


# ---------- 距离与代价 ----------

def Dis(C_1, C_2):
    """违反特征区间距离（anomaly distance），dist∈(-?,1]，越小越相符。"""
    if C_1.feature1 > C_2.feature1:
        if C_1.feature2 > C_2.feature2:
            return 1 - (C_2.feature2 - C_1.feature1) / (C_1.feature2 - C_2.feature1)
        else:
            return 1 - (C_1.feature2 - C_1.feature1) / (C_2.feature2 - C_2.feature1)
    else:
        if C_1.feature2 > C_2.feature2:
            return 1 - (C_2.feature2 - C_2.feature1) / (C_1.feature2 - C_1.feature1)
        else:
            return 1 - (C_1.feature2 - C_2.feature1) / (C_2.feature2 - C_1.feature1)


def Is_in(c, F_f):
    """找出可覆盖约束 c 的所有原因（解释）。"""
    waitlist = []
    for key in F_f:
        for cons in key.f_cover:
            if cons.name == c.name:
                waitlist.append(key)
    return waitlist


def Cost(c, X, F_f):
    """计算覆盖 c 的每条原因的解释代价 Cost(E)。"""
    waitlist = Is_in(c, F_f)
    costdic = {}
    cost = 0
    for item in waitlist:
        for key in item.f_cover:
            for cons in X:
                if cons.name == key.name:
                    cost = cost + Dis(cons, key)
        costdic[item] = cost
        cost = 0
    return costdic


def CalCost(H, X_all, F_f):
    """计算原因集合 H 的总原始代价。"""
    sumcost = 0
    for f_name in H:
        for cons in f_name.f_cover:
            for cons_X in X_all:
                if cons.name == cons_X.name:
                    sumcost = sumcost + Dis(cons_X, cons)
    return sumcost


def Inter(A, B):
    """两个集合元素名字相同则视为交元素。"""
    retSet = set()
    for x in A:
        for y in B:
            if x.name == y.name:
                retSet.add(x)
    return retSet


def Relation(f, rule_list):
    """关联比例 Relation(f) = 原因 f 覆盖的约束在频繁项中的最大占比。"""
    f_cover = set(f)
    maxratio = 0
    for rule in rule_list:
        if len(Set_Sub(f_cover, rule[0])) > 0:
            rule_0 = set(rule[0])
            rule_1 = set(rule[1])
            tempratio = len(Set_Inter(rule_0 | rule_1, f_cover)) / len(rule_0 | rule_1)
            if tempratio > maxratio:
                maxratio = tempratio
    return maxratio


# ---------- 原因选择 Select ----------

def Select(c, H, HasCovered, R, X, F_f, sumcost, rule_list,
           alpha=0.2, beta=4, gamma=0.1):
    """为约束 c 选择一条原因加入解集 H（逐行保持原型语义）。"""
    f_min = None
    if len(HasCovered) > 0:
        cost_min = Dis(list(HasCovered.keys())[0], c)
        f_min = HasCovered[list(HasCovered.keys())[0]]
        for item in HasCovered.keys():
            if cost_min > Dis(item, c):
                cost_min = Dis(item, c)
                f_min = HasCovered[item]
        if cost_min < gamma and f_min not in H:
            H.append(f_min)
            sumcost = sumcost + CalCost([f_min], X, F_f)
            for k in set(f_min.f_cover):
                for cons in X:
                    if cons.name == k.name:
                        X.remove(cons)
            if f_min in F_f:
                F_f.remove(f_min)
            return sumcost, H

    if c in X:
        TeCost = Cost(c, X, F_f)
        TeCost = sorted(TeCost.items(), key=lambda y: y[1])
        if len(TeCost) == 1:
            H.append(TeCost[0][0])
            sumcost = sumcost + TeCost[0][1]
            for k in set(TeCost[0][0].f_cover):
                for cons in X:
                    if k.name == cons.name:
                        X.remove(cons)
            F_f.remove(TeCost[0][0])
        elif len(TeCost) > 1:
            if len(Inter(TeCost[1][0].f_cover, X)) > len(Inter(TeCost[0][0].f_cover, X)):
                if Relation(TeCost[0][0].f_cover, rule_list) < Relation(TeCost[1][0].f_cover, rule_list) \
                        and TeCost[1][1] - TeCost[0][1] < alpha:
                    chosen = TeCost[1][0]
                elif Relation(TeCost[0][0].f_cover, rule_list) > Relation(TeCost[1][0].f_cover, rule_list) \
                        and len(Inter(TeCost[1][0].f_cover, X)) - len(Inter(TeCost[0][0].f_cover, X)) > beta \
                        and TeCost[1][1] - TeCost[0][1] < alpha:
                    chosen = TeCost[1][0]
                else:
                    chosen = TeCost[0][0]
            else:
                chosen = TeCost[0][0]
            # 统一收尾（与原型每个分支等价：加入 H、记 HasCovered、扣代价、从 X 移除其覆盖、从 F_f 删除）
            idx = 1 if chosen is TeCost[1][0] else 0
            H.append(chosen)
            HasCovered[c] = chosen
            sumcost = sumcost + TeCost[idx][1]
            for k in set(chosen.f_cover):
                for cons in X:
                    if cons.name == k.name:
                        X.remove(cons)
            F_f.remove(chosen)
    return sumcost, H


# ---------- 主算法 MyCover ----------

def MyCover(F_f, R, X, big_rule_list, alpha=0.2, beta=4, gamma=0.1, verbose=False):
    """集合覆盖主算法。返回 (sumcost, H)。

    F_f: 候选原因（解释）集；R: 待覆盖约束集；X: 检测出的约束违反特征集；
    big_rule_list: Apriori 关联规则；alpha/beta/gamma: 选择阈值。
    """
    Mincost = {}
    sumcost = 0
    H = []
    HasCovered = {}
    SortedK = sorted(X, key=lambda x: x.k, reverse=True)  # 按 k 值降序

    for i in range(len(SortedK)):
        if SortedK[i].k > 1:   # 先覆盖多序列约束(k>1)
            sumcost, H = Select(SortedK[i], H, HasCovered, R, X, F_f, sumcost,
                                big_rule_list, alpha, beta, gamma)
        else:
            X = SortedK[i:]    # 只留下 k==1 的
            break

    for c in R:                # 为每条约束计算最小代价
        Temcost = Cost(c, X, F_f)
        if Temcost:
            Mincost[c] = min(Temcost.values())
    SortedCost = sorted(Mincost.items(), key=lambda x: x[1])
    for tup in SortedCost:     # 按最小代价升序覆盖
        sumcost, H = Select(tup[0], H, HasCovered, R, X, F_f, sumcost,
                            big_rule_list, alpha, beta, gamma)

    if verbose:
        print(sumcost, end=', ')
        for f in H:
            print(f.name, end=' ')
        print("")
    return sumcost, H


if __name__ == "__main__":
    # 6 组演示数据（自原型 SetCovering.py 末尾搬入，验证与 legacy 输出一致）
    from . import demo_data
    rules = demo_data.demo_rules()
    for F_f in demo_data.demo_explanation_sets():
        MyCover(F_f, demo_data.demo_constraints(), demo_data.demo_constraints(), rules, verbose=True)
