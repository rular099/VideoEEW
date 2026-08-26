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

- 最终尾块已由上层 pipeline 显式 padding 并按有效帧裁剪；尚未在真实摄像头断流场景做板端验证。
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

## Phase G

### 完成内容

- 建立 94 条记录的数据清单：79 条直接配对、2 条 split sensor、13 条缺强震仪记录。
- 在 242 上完成 10 条真实视频、13,718 帧的完整 CoTracker/common/local/feature batch。
- 实现全强震记录 containment 搜索，同时比较加速度 proxy 与积分位移域，并明确多重搜索偏差。
- 实现 single-coefficient、Ridge、Huber、模型序列化、record/event group K-fold、PGA 分箱混淆矩阵与分组误差。
- 对齐相关低于 0.4 的记录 6、16、85 被显式拒绝进入研究 baseline。

### 修改文件

- `seismic_motion/pga/records.py`: 强震记录读取、PGA 定义和配对发现。
- `seismic_motion/pga/alignment.py`: 轴/符号/时间偏移和位移域候选对齐。
- `seismic_motion/pga/model.py`: 简单经验模型、group split 和误差指标。
- `scripts/run_pga_feature_batch.py`: 真实视频批量特征提取。
- `seismic_motion/cli/evaluate_pga.py`: 质量过滤和 group-CV 报告。
- `runs/20260823-pga-real/`, `runs/20260823-pga-evaluation/`: 紧凑证据。

### 关键设计决策

- 强震仪 horizontal-vector peak 是真值；视频二阶导数只作为输入特征，不冒充物理 PGA。
- 全记录最大相关只称 exploratory candidate；不声称已证明时钟同步。
- 预声明最简单单系数为 primary；复杂模型失败结果不隐藏。
- 缺失几何尺度时，离线 research-only 评估可运行，但保存模型仍 `requires_scale=true`，部署直接拒绝。

### 测试

- command: 242 真实 batch + 本地 `evaluate_pga` + unit/group-leakage/model round-trip tests。
- passed/failed: 10/10 视频完成；7/10 对齐候选通过阈值；33 项当前全量测试通过。
- metrics: primary MAE 94.4 gal，RMSE 118.1 gal，factor-2 71.4%，Spearman 0.464。

### 性能

- latency: 1709 blocks；mean 180.5、p50 160.2、p95 247.5、p99 445.5、max 6015.0 ms。
- memory: live batch snapshot peak RSS 1,267,432 kB，GPU memory 3906 MiB；不是逐 block 曲线。
- throughput: p95 小于 30 FPS 的 266.7 ms budget，但 p99/最大超限；共享 CPU load 约 98，不能验收实时性。

### 与 baseline 的数值差异

- 相比最初 ±2 s 假设，完整 containment 搜索在多条记录找到约 6–20 s 候选；记录 6 为 68.35 s，说明文件范围不等于事件起点近同步。
- 复杂 Ridge/Huber 在全 10 条未过滤敏感性运行发生严重外推；质量过滤后仍无有意义相关性，单系数更稳健。

### 未解决问题

- 只有 7 条可靠候选、同一未知 setup；无法评估跨相机/跨站点泛化。
- 几何尺度、camera/mounting/site metadata 和独立同步证据缺失；Milestone 4 仅通过“接口/真值/group split”，模型有效性未通过。

### 下一阶段建议

- 采集同步脉冲/共同触发、标尺与相机安装 metadata，再扩大按 event/camera/site 分组的数据量。

## Phase H

### 完成内容

- 完成 V100S PC baseline、真实 batch 资源快照与 10 分钟 bounded buffer/queue fast-forward。
- 实现三 worker 实时 runner、固定容量队列、过载停止、标准 timing/memory/queue/manifest 输出。
- 冻结 RK3588 realtime/low-compute 配置与板端采集字段。

### 修改文件

- `seismic_motion/runtime/pipeline.py`: 完整离线有界流水线和 provenance。
- `seismic_motion/runtime/realtime.py`: 实时捕获/跟踪/写盘与 silent-drop 防护。
- `benchmarks/runtime/streaming_stability.py`: 10 分钟逻辑快进稳定性。
- `reports/pc_stability.md`, `reports/rk3588_profiling.md`: 已测与阻塞边界。

### 关键设计决策

- 离线吞吐不能冒充 realtime；共享服务器压力结果也不作为可控验收。
- overload policy 是显式停止并拒绝 PGA，不任意丢帧破坏时间轴。

### 测试

- command: 18,000-frame fast-forward、真实 batch、realtime audit-contract unit test。
- passed/failed: bounded data-structure scope passed；RK3588 milestone blocked。
- metrics: buffer 最终 16，queue max/capacity 2/2，72 次拒绝全部计数，RSS slope 0.00185 MB/min。

### 性能

- latency: controlled rendered V100S p95 128.5–141.8 ms/block。
- memory: fast-forward 43.535 -> 44.125 MB；tracker history 由 reseed 限制到 528 帧。
- throughput: PC controlled rendered cases 满足 266.7 ms budget。

### 与 baseline 的数值差异

- 官方 online core 的历史 tensor 每 call 增长 8 帧；wrapper 将历史变为可审计固定上限。

### 未解决问题

- 无 RK3588 设备，无法记录 SoC/OS/governor/driver/temperature/throttling 或 30 分钟持续性能。

### 下一阶段建议

- 设备可用后严格按 `configs/rk3588_realtime.yaml` 运行 10/30 分钟验收。

## Phase I

### 完成内容

- 自动扫描本地 CoTracker source call chain，并导出固定 shape feature encoder ONNX opset 17。
- 运行 ONNX checker 与 PyTorch/ONNXRuntime 数值比较，生成带 shape inference 的实际算子表。
- 识别 full tracker 中 GridSample/Einsum 风险和 encoder 中 Resize/InstanceNormalization 转换审查项。

### 修改文件

- `seismic_motion/deployment/operator_audit.py`: source/ONNX 算子与形状清单。
- `scripts/export_feature_encoder.py`: encoder 边界导出与确定性 reference。
- `scripts/validate_feature_encoder_onnx.py`: ABI 隔离的 ONNXRuntime 验证。
- `reports/onnx_ops.csv`, `reports/rknn_compatibility.md`: 审计结果。

### 关键设计决策

- 242 共享环境 NumPy 2/ONNXRuntime ABI 不兼容，不修改共享环境；用只读 Python 3.10 NumPy 1.26 path 分进程验证。
- 未安装 RKNN Toolkit2 且无板，停止在 ONNX evidence，不声称 RKNN 支持。

### 测试

- command: fixed encoder export + ONNX checker + ONNXRuntime CPU inference。
- passed/failed: passed。
- metrics: max abs `1.997e-6`，mean abs `1.772e-7`，mean relative (`eps=1e-6`) `1.861e-5`。

### 性能

- latency/memory/throughput: 未做板端测量；ONNX 验证仅为数值验收。

### 与 baseline 的数值差异

- encoder ONNX 与 PyTorch 在固定输入上误差远小于已有 tracker subpixel RMSE；尚不能推导 full-track error。

### 未解决问题

- PyTorch 导出报告 InstanceNorm training-mode warning；虽 ONNXRuntime 数值通过，仍需真实 converter 检查。
- RKNN conversion log、FP16/INT8 feature error 和 full-track regression 均缺失。

### 下一阶段建议

- 在匹配 RKNN Toolkit2 的独立环境转换 encoder，并在真实板上逐层/端到端比较。

## Phase J

### 完成内容

- 冻结 encoder-NPU、sampling-CPU/Mali、correlation-MatMul、UpdateFormer-deferred 的分区配置。
- 在 production-like shape 上验证 Einsum -> batched MatMul 候选。
- 实现 synthetic/real-PGA 核心图、完整 audit-bundle generator 和 zip 输出。

### 修改文件

- `seismic_motion/deployment/correlation.py`: 等价 contraction 候选。
- `benchmarks/deployment/evaluate_correlation_rewrite.py`: 数值与 host timing。
- `configs/deployment_partition.yaml`: 分区成功定义和阻塞状态。
- `scripts/build_audit_bundle.py`, `scripts/plot_run.py`, `scripts/plot_pga_evaluation.py`: 审核交付。

### 关键设计决策

- 不追求 full-model 100% NPU；以吞吐、延迟、精度、稳定性为成功条件。
- MatMul 候选在进入 full tracker/ONNX/RKNN 前不替换上游代码。

### 测试

- command: local production-like correlation benchmark + audit bundle tests。
- passed/failed: FP32 tensor equivalence passed；full-track/RKNN blocked。
- metrics: max/mean abs/relative error 均 0；host mean 5.10 ms vs Einsum 5.78 ms。

### 性能

- latency: correlation host numbers only，不是 RK3588 estimate。
- memory/throughput: 板端未测。

### 与 baseline 的数值差异

- standalone correlation contraction exact；track output error 显式为 null，未错误外推。

### 未解决问题

- RK3588 Milestone 5、30 分钟稳定性和 thermal throttling 完全阻塞。
- GitHub 认证仍不可用，本地阶段提交尚未同步到 remote。

### 下一阶段建议

- 获取 RK3588 和 GitHub 凭据后，先 encoder RKNN，再 sampling fallback，最后 full-track regression 与持续实时验收。

## Next-stage status (Phases K-W)

本节覆盖 `1_cotracker3_rk3588_next_stage_codex_plan.md`，并取代上文中“GitHub
不可用”等已经过时的状态。GitHub `main` 已配置且允许阶段性自动推送。

### Phase K：因果信号链

- `OnlineSignalProcessor.update()` 逐样本输出 common/local 位移、速度、加速度和质量状态；
  causal SOS、导数历史和 running state 均为固定上限。
- backward finite difference 与只使用 `[t-N+1,...,t]` 的 causal polynomial
  derivative 均已实现，并有 future-invariance、非均匀时间戳和 bounded-state 测试。
- `configs/causal_realtime.yaml` 禁用 full-record detrend，记录 window=9、poly=3、
  filter order=4、startup policy 以及 signal lookahead=0。
- 27-case offline/causal 对比已冻结在
  `runs/20260823-causal-signal-benchmark/`。backward acceleration RMSE 为
  38.18 px/s²、peak amplitude bias 约 102%；causal polynomial RMSE 为
  77.08 px/s²、bias 约 229%。该失败结果说明因果导数尚不能直接支持物理 PGA 声明。
- 重要边界：CoTracker 16-frame window 输出前 8 个 source-time samples 时使用了
  8--15 帧未来视频上下文。因此 signal/PGA zero-lookahead 为 `PASS`，但当前
  end-to-end source-timestamp strict causality 为 `FAIL`。在线可用不冒充 Q2 通过。

### Phase L：Running PGA

- `RunningPGAEstimator` 同时保存 instantaneous estimate 与单调 running max。
- 无尺度时只输出明确标记的 pixel acceleration proxy；`pga_est` 物理部署字段为
  null，拒绝理由为 `deployment_output_rejected_without_geometric_scale`。
- Gal 映射接口仅允许显式 research mode；当前 `deployment_prediction_allowed=false`。

### Phases M、O、P：PGA evaluation v2、扩大样本与复杂度控制

- 固定输出 ALL、VIDEO-QUALITY-ONLY、POST-HOC-ALIGNED 三张逐记录表。
- ALL 与 VIDEO-QUALITY 的 selection function 只读取视频/运行时字段；单元测试通过
  改写真值和 alignment correlation 验证决策不变。POST-HOC 明确为不可部署诊断。
- 同时报告 median、single coefficient、log-linear、Ridge 和 Huber；复杂模型爆炸、
  外推和非有限风险不删除。primary 预声明为 single coefficient。
- bootstrap 固定 2000 次。event/camera/site metadata 保持 `UNKNOWN`，相应 group CV
  为 `NOT_EVALUABLE_METADATA_UNKNOWN`；record CV 为
  `PROVISIONAL_UNKNOWN_RECORD_RELATIONSHIPS`。
- 特征 schema 固定 `videoeew-motion-v1`，并另行强制检查 `causal=1`；evaluation 会拒绝
  schema 混用。早期 10-record offline input 仅作反例，`causal_pga_status=FAIL`：ALL
  single-coefficient MAE 112.64 gal、RMSE 133.73 gal；post-hoc 7 条 MAE 94.42 gal。
- 2026-08-25 在 242 启动全部 81 个 paired/split-paired 记录的因果批处理。record 0
  探针通过：470 blocks，tracker mean 180.96 ms、p95 191.65 ms；共享节点争用导致
  max 7866 ms。跨夜最终计数需在 SSH 恢复后取回，当前不标记完成。

### Phase N：alignment selection-bias audit

- circular-shift 与 phase-randomized surrogate 均已实现；每个 surrogate 重新执行完整
  axis/sign/lag/domain 最大搜索，而不是固定 observed candidate。
- 输出 empirical p-value 与 Benjamini-Hochberg q-value；正式 CLI 拒绝少于 1000 次。
- split sensor 使用全部分段拼接。真实数据正式 null run 尚未完成，状态 `NOT_MEASURED`。

### Phase Q：CoTracker stress benchmark

- coverage-designed matrix 覆盖 0.03--1.0 px、0.2--8 Hz、25/30/50/60 FPS、
  0.02--0.5 degree rotation，以及 rotation/local/occlusion/blur/illumination/interactions。
- 所有 case 调用真实 CoTracker adapter；oracle tracks 禁止作为模型误差。逐 case 保存
  point/common/local/rotation/causal-acceleration、survival、quality 与 tracker timing。
- 代码及矩阵 coverage tests 已通过；真实 GPU matrix 尚未执行，状态 `NOT_MEASURED`。

### Phase R：reseed boundary

- 已实现 point innovation、common translation/rotation jump、velocity jump、
  acceleration spike ratio、quality before/after 和每个 reseed 的 ±1 s 五联图。
- injected-jump unit test 通过。真实 81-record run 需待缓存完成后批量分析；未测前不判定
  reseed 是否会制造假峰值。

### Phases S、T：PC wall-clock realtime

- 三 worker runner 按 wall clock 读取 camera/file/looping playlist；每个 source/loop
  boundary 显式 reset tracker 与 signal state。
- 每 block 保存 capture/enqueue/tracker/motion/signal/PGA/write 时间戳，以及有界 queue、
  rejected item、RSS/peak RSS 与 overload event；标准 `runtime_*` 四张表已实现。
- 本机 CPU 10 s 冒烟保存在 `runs/20260825-realtime-local-cpu-smoke/`：第 32 帧
  capture queue 显式拒绝 1 帧并停止，首 tracker block 约 11.0 s。它验证 fail-closed，
  同时明确证明本机 CPU 不满足 30 FPS；短于 590 s，验收状态为 `NOT_TESTED`。
- 受 242 高负载和因果 batch 占用影响，受控 GPU 30 FPS 10/30 min 与 50 FPS run 尚未
  完成，均保持 `NOT_TESTED`。

### Phase U：尺度

- 当前无几何尺度。pixel-domain motion/proxy 可报告；Gal research mapping 明确
  `UNCALIBRATED/RESEARCH_ONLY`，deployment physical PGA 始终拒绝。

### Phases V、W：RKNN 与 RK3588

- 无可访问 RK3588；本阶段没有 converter log、板端数值误差、latency、memory、温度、
  throttling 或 10/30 min 数据。
- encoder ONNX 的既有 host 数值证据保留，但 RKNN conversion 与 RK3588 realtime 均为
  `BLOCKED_NO_DEVICE`，不得写为预测通过。

### Audit Bundle v2 与测试

- composite assembler、固定图名生成器和 bundle builder 已实现。bundle 缺项写为
  `NOT_MEASURED/NOT_TESTED/NOT_EVALUABLE/BLOCKED`，不会生成虚构数值。
- `AUDIT_SUMMARY.md` 直接回答计划要求的 A--L；区分 signal causality、tracker
  source-time causality 与 online availability。
- 当前本地全量测试：53 passed / 0 failed（2026-08-26，
  `python -m unittest discover -s tests -v`）。
