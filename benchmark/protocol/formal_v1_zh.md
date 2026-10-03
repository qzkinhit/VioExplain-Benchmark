# VioExplain 正式基准协议 v1

本协议用于 2026-10-03 新运行。它不把此前 12 文件 SKAB 小试或使用真值解释大小的历史试验改称正式全量结果。

## 任务与数据

|轨道|数据版本与范围|输入监督|输出与评价|
|---|---|---|---|
|原生检测|SMD 官方全部 28 台，官方 train/test|训练文件为正常参考，测试标签仅评分|原始逐点 AUPRC/AUROC、正常校准阈值后的 F1/FPR、事件命中与延迟|
|条件维解释|SMD 全部官方 interpretation_label，组1开发，组2/3最终评价|给定事件区间；不输入真值变量集合大小|固定 K=1/3/5 的 P/R/F1、NDCG、Hit，逐事件记录|
|原生检测|SKAB commit b2c0d46c2971dcbfe71e26087b6d231998bb91c2 全部34异常CSV|独立 anomaly-free 文件正常训练/校准|与SMD同检测指标，宏平均按CSV；不把检测标签当逐维根因|
|原生检测|MIT/Braatz经典TEP全部d00..d21测试文件|仅d00.dat正常训练/校准|0正常与21故障全部保留，test故障按经典分发约定从0-based160起，该标记由协议构造，并非独立的官方标签文件；不剔除难故障|
|事件类解释|相同TEP的独立官方故障train/test文件|历史故障类别1..21为知识；训练内部划分后再测试|F01..F21事件候选解释，须单独申明窗口和训练标签量|

TEP 是化工过程模拟，不是真实工厂检修数据。官方 readme 的52列顺序为 XMEAS(1..41)、XMV(1..11)。d00.dat 实际为52×500，适配器明确转置；其余故障训练480×52，测试960×52。不得复用本地其他项目错误的22测量+30操控列名。官方 teprob.f.txt 为1..15给出物理故障说明，16..20标为Unknown；21号只使用编号，未取得可靠描述前不编机理名称。经典版每类仅一条训练轨迹和一条测试轨迹，分窗不是新增独立实验。

## 隔离与固定设置

所有正常训练文件按时间前60%拟合，60–80%预留尺度/开发，末20%校准。统计方法直接使用末20%校准原分数，不因未使用中段而把它加入训练。阈值为正常校准分数的有限样本0.99上次序统计量；时序相关性下不据此声称无条件1%误报保证。模型拟合、归一化和阈值均不能查看测试标签或测试异常比例。

主检测结果逐点评价，不使用 point adjustment，不用测试最佳阈值。测试单类别文件的AUPRC/AUROC记为不可用而非人为填0。正常误报率和事件延迟独立列出；未命中事件不记为零延迟。

维解释是给定异常区间的条件评价。SMD全38维供每种排序基线排序。BARO用事件前等长已观测区间作参照，若起点为0则用正常训练尾部；不检查参照是否真正常，不依据标签挑参照。统计与TranAD用正常训练拟合的全维分数。TreeSHAP对每事件至多30个等距点归因，不按异常点标签筛样。主K固定，不读取真值集合大小。

随机模型使用种子0、1、2。确定性PCA/Robust z/BARO只运行一次，seed=-1，不把复制结果当独立重复。检测按整台机器/整条轨迹宏平均，解释按事件平均后另按机器重采样。跨种子的均值与标准差不能当跨工厂泛化置信区间。

## 权威方法与适配边界

|方法|固定来源|状态/忠实范围|
|---|---|---|
|BARO RobustScorer|phamquiluan/baro e35f4ec1095e5cac891d52de9ad18a5b32a37ec8，MIT|直接调用官方DataFrame RobustScorer；未运行BOCPD，不称完整BARO链路|
|TreeSHAP + Isolation Forest|shap 0.46.0，官方TreeExplainer；sklearn IsolationForest 200树|训练段拟合，path-dependent TreeSHAP解释路径长度；不是因果解释，也不冒称原历史KernelSHAP结果|
|TranAD|imperial-qore/TranAD 7ffb98d0c18189cc3d9ab732b4cb0278200a0af0，BSD-3-Clause|官方网络类与两阶段损失不变，5 epoch、float64、128 batch、AdamW lr1e-4、weight_decay1e-5、StepLR(5,.9)|
|SARAD|daidahao/SARAD 24854d9723b4eed31b547344061671c08fbfb3e2，MIT|SMD全部28台、三种子运行完成；共享模型、机器边界与尺度校准适配见docs/SARAD_ADAPTATION_zh.md|
|GDN|d-ailin/GDN 官方仓库|已核论文/官方代码；本轮未运行，不填虚构分数|

TranAD官方脚本仅硬编码machine-1-1，正式适配遍历全部28台。网络类以AST提取以避开不相干DGL及全局argparse导入。PyTorch2会向旧自定义层传新causal参数，适配使用与原一层容器计算一致的顺序调用，不改变网络层。官方window在索引i重建i-1，正式输出把分数对齐被重建观测t；窗口仅含≤t观测。推理分批限制显存。仅用fit段min-max变换所有分区，不独立拟合测试归一化，越训练范围值不裁剪。统一正常校准阈值替代官方POT/PA评分。这些均记录在run manifest，不把新协议分数与论文表格直接比较。

AEC-Prototype仅指所提供旧原型的MyCover与Select，尚未据权威最终博士稿完成一致性核验。MinExplain是opening+assignment扩展目标。二者分别标识，不能把扩展求解目标的改善表述为原算法复现修复。历史 publicexp 相关约束缺截距的问题另记录，不在保持历史结果的同时暗改实现。

## 结果工件

每次运行保存源文件hash、数据文件hash、环境版本、配置、逐文件检测、逐事件解释、预测score文件与阈值，训练模型额外保存checkpoint hash和逐epoch loss。失败输出保留并标记，修复重跑写新目录。COMPLETE.json只在所有配置对象处理完后写入。所有研究实验在远端Linux服务器运行，本地仅编辑、同步和读取结果。

官方入口：[SMD](https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset)、[SKAB](https://github.com/waico/SKAB)、[TEP](https://web.mit.edu/braatzgroup/links.html)、[BARO](https://github.com/phamquiluan/baro)、[TranAD](https://github.com/imperial-qore/TranAD)、[SARAD](https://github.com/daidahao/SARAD)、[SHAP](https://github.com/shap/shap)。

## TEP 起点与事件分类补充


测试故障开始于零基索引160采用经典发布版约定。官方压缩包的 `temain_mod.f.txt` 设置 `NPTS=172800`、`SSPTS=3600*8`、`DELTAT=1/3600` 小时，并每180次积分写一次观测，即48小时、3分钟采样、8小时切入。该driver保留故障12示例，不声称每个dXX文件都带完整独立生成记录。数据并不包含单独标签文件。CMU Kitchin研究组的[公开复现](https://kitchingroup.cheme.cmu.edu/tep-rust/notebooks/02-fault-detection-pca.html)也明确对应发布的d00..d21采用160点正常前缀。

TEP条件分类使用64点完整非重叠窗。故障训练段从20开始到floor(.6N)，开发段[.6N,.8N)，校准段[.8N,N)，各窗不跨段。正常类0从0开始同规则。额外删除故障训练前20点是预设warm-up exclusion，不声称原始480点训练文件在20处才出现故障。[Chiang等2015作者稿](https://web.mit.edu/braatzgroup/diagnosis_of_multiple_and_unknown_faults_using_the_causal_map_and_multivariate_statistics.pdf) §3.1说明500点生成记录只取故障后的480点用于模型构建，进一步提示不能混淆经典480点文件和Rieth500点版本。

条件分类的测试故障窗从160开始，正常轨迹从0开始，共267窗、22轨迹。末尾不足64点不用作分类窗，但仍包含于逐点检测。按窗口和整轨迹投票分别评分，平票选择最小类别号，不能把同轨迹的窗口视为独立复现。

强监督分类基线为RBF-SVM、自动收缩LDA、500树RandomForest。特征为每通道均值、标准差和线性斜率，全部从64点窗计算。SVM/LDA标准化只拟合训练窗；SVM在开发窗选C∈{0.1,1,10,100}与gamma∈{scale,.001,.01}，平分保留列表最先项。LDA为lsqr+shrinkage=auto。RF用0/1/2种子，其他确定性方法一次。校准窗保留但这些闭集分类器不使用它；不在看到测试分数后重拟合模型。
