# coding=utf-8
"""关联规则挖掘（Apriori）—— 用于计算违反特征间的频繁项集与关联规则。

重构自原型 legacy/setcovering/Apriori.py，算法逐行保持，仅：
 1. 复用 models.Constraint（删去重复类定义）；
 2. 修复原型 bug：Dis 中 `C_1.featrue1` → `C_1.feature1`（拼写错误）。

数据集形式：以一定长度时间段内的约束违反为一个列表元素，列表中记录该时段每个约束的违反特征。
"""
from .models import Constraint  # noqa: F401  (供外部 from apriori import Constraint 的兼容)


# 计算距离（注：原型此函数有拼写 bug 且未被主流程调用；此处修正 featrue1→feature1）
def Dis(C_1, C_2):
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


# 1-频繁项集
def Create_C1(dataSet):
    C1 = set()
    for t in dataSet:
        for c in t:
            C1.add(frozenset([c]))
    return C1


def Is_Apriori(Ck_item, Lksub1):
    for item in Ck_item:
        sub_Ck = Set_Sub(Ck_item, frozenset([item]))
        for Lk in Lksub1:
            if Equal_List(list(sub_Ck), list(Lk)):
                return True
    return False


# 判断两个存储约束违反的列表是否相等
def Equal_List(list1, list2):
    for i in range(len(list1)):
        if not list1[i].similar(list2[i]):
            return False
    return True


# 判断两个 set 是否相等
def Equal_Set(set1, set2):
    if len(set1) != len(set2):
        return False
    flag = 0
    for x in set1:
        for y in set2:
            if x.similar(y):
                flag += 1
                break
    return flag == len(set1)


# set1 是否是 set2 的子集
def SubSet(set1, set2):
    flag = 0
    for x in set1:
        for y in set2:
            if x.similar(y):
                flag += 1
                break
    return flag == len(set1)


# 集合差 set1 - set2
def Set_Sub(set1, set2):
    set1 = set(set1)
    set2 = set(set2)
    for x in set1.copy():
        for y in set2:
            if x.similar(y):
                set1.remove(x)
    return set1


# 集合交 set1 ∩ set2
def Set_Inter(set1, set2):
    retSet = set()
    for x in set1:
        for y in set2:
            if x.similar(y):
                retSet.add(x)
    return retSet


# 产生 Ck，即 k-频繁项集候选
def Create_Ck(Lksub1, k):
    Ck = set()
    list_Lksub1 = list(Lksub1)
    len_Lksub1 = len(Lksub1)
    for i in range(len_Lksub1):
        for j in range(1, len_Lksub1):
            l1 = list(list_Lksub1[i])
            l2 = list(list_Lksub1[j])
            l1.sort(key=lambda x: x.name)
            l2.sort(key=lambda x: x.name)
            if Equal_List(l1[0:k - 2], l2[0:k - 2]) and l1 != l2:
                Ck_item = frozenset(list_Lksub1[i] | list_Lksub1[j])
                if Is_Apriori(Ck_item, Lksub1):
                    Ck.add(Ck_item)
    return Ck


# 通过删除策略从 Ck 产生 Lk
def Generate_Lk_by_Ck(dataSet, Ck, min_support, support_data):
    Lk = set()
    temp = None
    item_count = {}
    for t in dataSet:
        for item in Ck:
            if SubSet(item, t):
                flag = 0
                for key in item_count.keys():
                    if Equal_Set(item, key):
                        flag = 1
                        temp = key
                        break
                if flag == 0:
                    item_count[item] = 1
                else:
                    item_count[temp] += 1
    t_num = float(len(dataSet))
    for item in item_count:
        if (item_count[item] / t_num) >= min_support:
            Lk.add(item)
            support_data[item] = item_count[item] / t_num
    return Lk


# 生成所有频繁项集
def Generate_L(data_set, k, min_support):
    support_data = {}
    C1 = Create_C1(data_set)
    L1 = Generate_Lk_by_Ck(data_set, C1, min_support, support_data)
    Lksub1 = L1.copy()
    L = [Lksub1]
    for i in range(2, k + 1):
        Ci = Create_Ck(Lksub1, i)
        Li = Generate_Lk_by_Ck(data_set, Ci, min_support, support_data)
        Lksub1 = Li.copy()
        L.append(Lksub1)
    return L, support_data


# 从频繁项集生成关联规则
def Generate_big_rules(L, support_data, min_conf):
    big_rule_list = []
    sub_set_list = []
    for i in range(0, len(L)):
        for freq_set in L[i]:
            for sub_set in sub_set_list:
                if SubSet(sub_set, freq_set) and len(set(freq_set) - set(sub_set)) > 0:
                    temp = frozenset(set(freq_set) - set(sub_set))
                    conf = support_data[freq_set] / support_data[temp]
                    big_rule = (Set_Sub(freq_set, sub_set), sub_set, conf)
                    if conf >= min_conf and big_rule not in big_rule_list:
                        big_rule_list.append(big_rule)
            sub_set_list.append(freq_set)
    return big_rule_list
