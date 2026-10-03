# coding=utf-8
"""集合覆盖解释算法的数据模型。

统一原型中 SetCovering.py 与 Apriori.py 各自重复定义的 Constraint / Reason 类。
语义与原型完全一致：
 - Constraint = 一条约束的违反特征 violation feature ⟨name, [feature1, feature2]⟩，
   name 对应约束/序列标识，[feature1,feature2] 为违反程度区间，k=约束涉及序列数，w=可能性权重。
 - Reason     = 一条异常解释 anomaly explanation / 原因，name + f_cover（其覆盖的约束违反特征集）。
"""

# 相似判定阈值 THRESHOLD（原型中为模块级常量 2）；保留为可配置模块变量。
THRESHOLD = 2


def set_threshold(value):
    """供配置注入 similar/__eq__ 的阈值。"""
    global THRESHOLD
    THRESHOLD = value


class Constraint:
    """约束违反特征 violation feature（逐字沿用原型 SetCovering.Constraint 语义）。"""

    def __init__(self, name, feature1, feature2, k=1, w=1):
        self.name = name
        self.feature1 = feature1
        self.feature2 = feature2
        self.k = k          # 约束涉及的序列条数（多序列约束 k>1）
        self.w = w          # 可能性表征的出现概率权重

    def similar(self, other):
        return (abs(self.feature1 - other.feature1) < THRESHOLD
                and abs(self.feature2 - other.feature2) < THRESHOLD
                and self.name == other.name)

    def __eq__(self, other):
        return self.similar(other)

    def __hash__(self):
        # 原型用 ord(name)，要求 name 为单字符；保留以维持 set/dict 行为一致。
        # 多字符 name 退化为按首字符哈希（仍满足 hash 一致性约束）。
        return ord(self.name[0]) if self.name else 0

    def __repr__(self):
        return f"C({self.name},{self.feature1},{self.feature2},k={self.k})"


class Reason:
    """异常解释 / 异常原因 anomaly explanation。"""

    def __init__(self, name, f_cover):
        self.name = name
        self.f_cover = f_cover   # 该原因覆盖的约束违反特征集合（set/list of Constraint）

    def __repr__(self):
        return f"R({self.name})"
