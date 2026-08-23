# Implementation status

## Phase A

### 完成内容

- 冻结本机及 242 的 CoTracker、Python、PyTorch、CUDA 和主要依赖状态。
- 记录本地 CoTracker commit、dirty 状态、checkpoint SHA-256、关键算子和在线缓存。
- 在 242 的 V100S GPU 3 上重复运行官方 16 帧 online predictor。
- 建立版本化 YAML 配置、配置校验、环境/文件哈希工具和标准 run 目录。
- 初始化 VideoEEW Git 仓库；本地首提交已创建。

### 修改文件

- `configs/pc_baseline.yaml`: 第一版完整、非硬编码默认配置。
- `seismic_motion/config.py`: 配置读取、跨字段校验和确定性哈希。
- `seismic_motion/diagnostics/provenance.py`: 环境、Git 和 SHA-256 记录工具。
- `reports/cotracker_local_inspection.md`: 本地源码、缓存和算子审计。
- `reports/baseline_environment.md`: 本机/242 可复现环境与官方基线。
- `runs/20260823-phasea-official-baseline/`: Phase A 原始审计记录。

### 关键设计决策

- 不修改或清理用户的 CoTracker checkout；通过外部路径 adapter 调用。
- 242 共享 conda 环境不安装新包；缺失依赖进入项目私有环境。
- 官方在线实现的历史预测 tensor 线性增长，Phase B 使用有界、显式记录的周期 reseed。
- 无几何尺度时拒绝物理 PGA 输出，只保留像素域特征。

### 测试

- command: `PYTHONPATH=. python -m unittest discover -s tests -v`
- passed/failed: 2 passed / 0 failed
- metrics: 同输入重复轨迹最大差异 0 px，visibility disagreement 0。

### 性能

- latency: V100S warm-up 585.601 ms，warm 139.482 ms / 16-frame call。
- memory: 尚未执行完整管线内存曲线；Phase B 后测量。
- throughput: warm tracker call 低于 266.7 ms 的八帧到达预算。

### 与 baseline 的数值差异

- 当前阶段直接调用官方 predictor，没有算法改写，重复运行差异 0 px。

### 未解决问题

- GitHub HTTPS/SSH 均无可用凭据，首提交暂未推送；本地 commit 安全保留。
- 视频与强震仪的精确偏移未知，待视觉信号生成后估计。
- RK3588 不可访问；板端数据不得推断或伪造。

### 下一阶段建议

- 实现稀疏点 adapter、有界 frame/block buffer、周期 reseed 和标准化轨迹输出。

