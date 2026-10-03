# SARAD 官方适配审计

官方源为 [daidahao/SARAD](https://github.com/daidahao/SARAD)，commit `24854d9723b4eed31b547344061671c08fbfb3e2`，MIT许可。运行前验证四个源文件的SHA-256。核心网络类从官方sar73.py和sar76.py的AST原样载入，只避免与网络无关的Lightning、Hydra和日志导入。

依据官方 configs/data/default.yaml、configs/data/smd.yaml、configs/model/sar76.yaml、configs/trainer/default.yaml、configs/experiment/sar76.yaml 及callback最终覆盖，采用共享SMD模型、窗长100、batch128、随机10%训练位置、3epoch、512维3层8头、两个时间patch、64维关联重建头、dropout0.1、Adam lr0.0007、每epoch学习率减半。没有提前停止，取最后epoch。运行3个独立种子0、1、2，测试覆盖全部28机器。

保留官方训练损失 `mean((x_hat-x)^2) + 100*mean((q_bar-q)^2)`。q来自分离梯度的关联变化，所以关联重建梯度只进入检测头；这一点由未改的官方forward保留。检测分数为标准化重建误差与标准化关联重建误差之和；维诊断为两种逐维标准化误差各0.5权重。没有把单一重建分数冒充SARAD。

统一协议必须做三项明确适配。第一，正常轨迹按前60%拟合、中20%估计官方分数均值/标准差、末20%固定1%名义阈值，全部在测试前完成。第二，官方val_dataloader实际从data_train随机取样而未用data_val，本适配显式使用隔离中段；不复制这一验证集实现错误。第三，官方数据集先拼接机器再切窗可能跨机器边界，本适配分别构窗，避免把无关机器末尾当当前机器历史。

StandardScaler拟合全部28机器各自前60%正常数据，符合官方共享模型的归一化形式；测试使用相同变换。每个测试时间点的100点窗仅用已观测历史与当前点，起始不足窗长复制首点。阈值按机器在正常校准末段独立确定。报告原始逐点评分，无point adjustment。所有327个官方解释事件都有结果，组2/3的216个事件用于最终条件维解释。共享模型不使用任何解释标签训练。

训练预算、采样位置、逐epoch损失、正常尺度、checkpoint与SHA、逐机器score/diagnosis和逐事件排名均保留。模型权重中归一化buffer未用于另行推断；重放应载入对应 `seedN_normal_scales.npz`，它保存全部评分尺度。
