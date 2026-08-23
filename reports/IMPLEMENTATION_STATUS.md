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

## Phase B

### 完成内容

- 实现手工点与分区 Shi–Tomasi 自动角点，避免查询点集中在单一区域。
- 实现标准 `TrackBatch`，保存 timestamp、frame/point id、xy、visibility、query 点和 reseed id。
- 实现固定窗口 `SlidingFrameBuffer` 与满载时显式拒绝的 bounded queue。
- CoTracker adapter 只输出已稳定的八帧；EOF overlap 由 `flush_pending()` 输出。
- 通过周期 reseed 将公开 online core 的历史 tensor 上限固定，并记录每次 reseed。

### 修改文件

- `seismic_motion/tracking/cotracker_adapter.py`: 稀疏 online adapter 与有界 reseed。
- `seismic_motion/tracking/online_buffer.py`: frame ring buffer 和审计队列。
- `seismic_motion/tracking/point_selection.py`: 手工/自动分布式查询点。
- `seismic_motion/tracking/types.py`: 标准轨迹协议。

### 关键设计决策

- 默认每 64 个 block reseed，上游历史上限 528 帧；reseed 使用 overlap 中的最新点坐标保持 point id。
- predictor 不提供原始 confidence，因此 confidence 保存为 NaN，不用布尔 visibility 冒充置信度。
- 上游 predictor 硬编码六次迭代；四次迭代在实现数值审计 backend 前显式拒绝。

### 测试

- command: 242 GPU 3 两窗口 adapter integration test。
- passed/failed: passed；输出帧分别为 0–7、8–15，shape 均为 `[8,32,2]`。
- metrics: 强制 reseed 事件 1 次；输出坐标全部有限；visibility disagreement 0（synthetic cases）。

### 性能

- latency: V100S rendered benchmark warm mean 125–134 ms/block，p95 129–142 ms。
- memory: 上游单周期理论历史上限 528 帧；实际长稳 RSS 测试留待 PC 验收阶段。
- throughput: warm p95 低于 266.7 ms 到达预算。

### 与 baseline 的数值差异

- adapter 不改模型数学；它切出稳定八帧并在达到上限时显式 reseed。
- forced reseed 的连续性已验证为有限输出，长期 reseed 边界误差将在长稳测试统计。

### 未解决问题

- 不足 16 帧的最终尾块尚由上层 pipeline 负责显式 padding/有效帧裁剪。
- GitHub 认证仍缺失。

### 下一阶段建议

- 用 synthetic 已知运动量化 tracker 与 common/local 分解。

## Phase C

### 完成内容

- 实现 pure translation、translation+local、rotation、组合运动及遮挡/模糊/光照变化生成器。
- 覆盖 0.1–5 px、0.5–8 Hz、25/30/60 FPS 及 0.05–1° 旋转。
- 完成 116-case oracle matrix 和两组真实 CoTracker rendered benchmark。

### 修改文件

- `benchmarks/synthetic/spec.yaml`: 完整参数矩阵。
- `benchmarks/synthetic/generator.py`: subpixel frame 与 point truth 生成器。
- `benchmarks/synthetic/evaluate.py`: oracle common/local regression。
- `benchmarks/synthetic/evaluate_cotracker.py`: rendered CoTracker 数值评估。

### 关键设计决策

- oracle matrix 只证明 geometry/signal 实现，不作为 CoTracker accuracy 证据。
- CoTracker benchmark 单独报告，以防把 exact tracks 的零误差误称为模型性能。

### 测试

- command: `python benchmarks/synthetic/evaluate.py ...` 及 242 rendered benchmark。
- passed/failed: 116 oracle cases + 2 rendered cases passed。
- metrics: pure translation CoTracker RMSE 0.0942 px；组合 case RMSE 0.4368 px。

### 性能

- latency: rendered warm p95 141.8 ms（translation）、128.5 ms（combined）。
- memory: synthetic ground truth 以 NPZ 保存但不进入 Git。
- throughput: 两 case 均满足 V100S 八帧预算。

### 与 baseline 的数值差异

- pure translation amplitude error 0.0343 px；combined amplitude error 0.0240 px。
- combined local residual RMSE 0.3103 px。

### 未解决问题

- 尚未覆盖 CoTracker 的全部 116-case 图像渲染矩阵，完整矩阵成本留给批量 benchmark。

### 下一阶段建议

- 在真实视频上检查 similarity/affine 稳定性。

## Phase D

### 完成内容

- 实现 translation、similarity、affine 和可选 homography 的确定性拟合。
- 实现 deterministic RANSAC、inlier mask、fit RMSE、coverage、condition number 与跳变门控。
- 保存 common transform、原始 tracks 和逐点 local residual。

### 修改文件

- `seismic_motion/motion/global_motion.py`: 模型拟合/RANSAC/矩阵分解。
- `seismic_motion/motion/quality.py`: GOOD/DEGRADED/INVALID 门控。
- `seismic_motion/motion/residual_motion.py`: sequence common/local 分解。

### 关键设计决策

- 默认 similarity；homography 仅为可选数值模型，不解释为 3D pose。
- low coverage 可降级，tracks/inliers/RMSE/condition 严重不足则 INVALID。

### 测试

- command: unit tests + 116-case oracle matrix。
- passed/failed: passed。
- metrics: oracle 最大 common point RMSE `4.08e-6 px`，最大 rotation RMSE `2.86e-7°`。

### 性能

- latency: 尚未从 tracker timing 中独立拆出；full pipeline profile 阶段补充。
- memory: frame-wise arrays 与输入长度线性，但运行时分块写盘；不进入无界 queue。
- throughput: 非当前瓶颈。

### 与 baseline 的数值差异

- exact synthetic residual 最大 RMSE `4.21e-6 px`。

### 未解决问题

- 真实场景多数点属于运动物体时仍可能错误选择 common consensus，需由 quality/ROI 策略拒绝。

### 下一阶段建议

- 将相机/场景 metadata 纳入质量审计。

## Phase E

### 完成内容

- 实现 known-length `mm_per_px` 标定、ROI 有效性检查和完整 metadata。
- 实现 `UNCALIBRATED` 状态，缺失尺度时 mm 转换直接拒绝。

### 修改文件

- `seismic_motion/calibration/scale.py`: 明确的尺度状态与转换。

### 关键设计决策

- 当前数据缺少几何尺度，所有现有 run 均标记 `UNCALIBRATED`。
- rotation 永不乘单一 `mm_per_px`；不同深度不共享未经验证的尺度。

### 测试

- command: scale unit tests。
- passed/failed: 2 passed。
- metrics: 100 mm / 250 px 精确转换为 0.4 mm/px；未标定转换被拒绝。

### 性能

- latency: 可忽略。
- memory: 可忽略。
- throughput: 非瓶颈。

### 与 baseline 的数值差异

- 无物理 baseline；未生成伪造 mm 结果。

### 未解决问题

- 等待真实标尺或平面几何参考。

### 下一阶段建议

- 后续采集在监测平面放置已知长度参考物。

## Phase F

### 完成内容

- 实现 timestamp 诊断、missing-frame 估计和 uniform resampling。
- 实现 causal SOS 与 offline zero-phase bandpass，并在 metadata 中区分。
- 实现真实时间戳 finite difference 与 local polynomial 一/二阶导数。
- 实现 common/local/quality window features 和 signal benchmark。

### 修改文件

- `seismic_motion/signal/timestamps.py`: 时间轴审计与重采样。
- `seismic_motion/signal/filtering.py`: causal/offline filter。
- `seismic_motion/signal/derivatives.py`: 两类 timestamp-aware 导数。
- `seismic_motion/signal/features.py`: versioned feature vector。
- `benchmarks/signal/evaluate.py`: 噪声、peak bias 与 lag 比较。

### 关键设计决策

- offline zero-phase 只用于评估，绝不标记 realtime。
- raw second difference 仅保留为反例 baseline。
- 频带 0.3–8 Hz 是配置起点；不声明为固定物理真理。

### 测试

- command: signal unit tests + 27 cases/method signal benchmark。
- passed/failed: passed。
- metrics: raw second-difference median RMSE 35.45 px/s²；offline bandpass + local polynomial 为 6.83 px/s²。

### 性能

- latency: offline 方法零相位但非实时；causal 方法 median lag 1 sample，极端低 SNR case 明显恶化。
- memory: causal filter 仅保留固定 SOS state；local polynomial 运行时仅需固定窗口。
- throughput: 非 tracker 瓶颈。

### 与 baseline 的数值差异

- signal benchmark 证明直接二阶差分显著放大 0.02 px 噪声。

### 未解决问题

- causal filter 在低振幅/高频 case 的 p95 误差较大，必须由真实强震数据调参，不能直接用于 PGA 声明。

### 下一阶段建议

- 配对强震仪记录，估计时间偏移并建立 group-split PGA baseline。
