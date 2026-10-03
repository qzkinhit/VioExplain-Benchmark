# coding=utf-8
"""公开真实数据评测的方法侧代码（协议 docs/protocol_public.md 的实现层）。

纪律：
 - 三隔离（约束挖掘段 / 知识构建段 / 评测段互不重叠），任何方法拿不到它评测事件的根因标注。
 - 本包只做「方法」：约束挖掘、违反特征、AEC 解释、事件级评测、E-G 检测器接口。
   数据 IO 走 tools/standardize_public.py（工具层）；结果落盘走 experiments/exp_public_*.py。
 - 不改 src/vioexplain/mincost 与 setcover（其它代理在用）：本包只 import 复用其纯函数。

模块：
 - constraints  : 从 train 段挖 domain/speed/corr 约束界（协议 §1.2），逐点检测违反。
 - features      : 事件窗口 -> 违反特征（V_dims 与逐条量化区间），喂给所有方法同一违反空间。
 - knowledge     : 从训练组事件学异常表征知识集 R（协议 §1.3，跨组，不看真值）。
 - aec           : 违反 + 知识 -> MinExplain 实例 -> greedy 解释 -> 映射回维集合（协议 §1.4）。
 - evaluate      : 事件级根因维 P/R/F1（micro/macro），点级 F1（PA / 非 PA），manifest。
 - encoding      : E-G 违反特征编码（逐点逐维越界指示），供检测器 ± 拼接。
"""
