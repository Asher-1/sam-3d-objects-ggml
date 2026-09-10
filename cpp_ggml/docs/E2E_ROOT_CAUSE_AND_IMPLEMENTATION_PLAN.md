# 全面 E2E 审查与剩余实施方案

审查日期：2026-09-10。最新指标只维护在 [完整 GLB 报告](../benchmarks/e2e_comparison/full_glb_current/README.md) 和 [机器可读矩阵](../benchmarks/e2e_comparison/e2e_latency_current.json)，不再把历史 image-to-PLY 耗时当作完整重建速度。

**功能链路已连接，不等于官方精度、速度和稳定性验收通过。** 本轮实际执行官方 Python、CUDA F16/Q8_0/Q4_0、Vulkan F16/Q8_0/Q4_0 推理接 CUDA 后处理的 raw 重建，并保存 GLB、独立 PNG、pose、60 视图对比和失败门禁。没有进行新的性能优化、QAT 或降低烘焙规格。

## 1. 比较对象与证据边界

```text
image + mask + native seed
  -> MoGe / conditioning / SS diffusion / occupancy / support / pose
  -> SLat diffusion / Gaussian / FlexiCubes mesh
  -> cleanup / xatlas UV / 100 Gaussian observations
  -> 2500 Adam + TV updates / Telea / material / GLB + base-color PNG
```

Vulkan 只负责神经推理，独立 CUDA 进程负责 PBR。交接的 PLY、原始顶点、面、pose 和 dtype 收据必须在交接前后保持 SHA-256 不变。原生路径没有 Python、Torch 或 cuDNN 推理回退；官方 Python 仅为另一种启动方式和独立 oracle。

同 seed 不自动意味着同噪声，F16 不等于 BF16，Q4 文件小不自动意味着更快，GLB 可以打开也不意味着材质一致。每个差异要区分“本轮实测”“源码确认”和“尚待验证的归因”。不得减少扩散步数、可见性视图、100 个烘焙视图、2500 次更新或 1024px atlas 来获得同任务加速结论。

| 维度 | 本轮采用的边界 | 原因与限制 |
| --- | --- | --- |
| 入口 | 根目录两个薄启动器，共用参数验证和既有实现 | 不另造推理框架；拒绝覆盖非空输出目录 |
| 内存 | 官方分阶段驻留；原生按阶段加载 GGUF | 在 12 GiB 设备运行，不声称所有模型常驻 |
| 时间 | 进程启动到 GLB/PNG 写入完成并退出 | 包含导入、初始化、加载和后处理；外部评分不计入 |
| 数值 | fresh stage oracle 与实际 dtype manifest | 官方使用受限显存 mixed policy，不伪称原始纯 F16 |
| 后端 | CUDA 完整进程；Vulkan 推理加独立 CUDA PBR | 不要求也不宣称 Vulkan PBR 实现 |
| 发布 | 实际资产、哈希、单 patch 重放、失败闭合门禁 | 不用测试代码通过替代模型精度通过 |

## 2. 已实施及本轮验证

- 根目录 `run_python.sh`、`run_ggml.sh` 输出含内嵌 base-color PNG 的 GLB、独立纹理图和 pose JSON。参数、缺失模型、错误路径、非空输出目录、构建许可证开关均有明确检查。
- `session.cpp` 的耗时日志改用单调墙钟；原生 CUDA 和 Vulkan 接 CUDA 均实际生成了全部三种精度的完整资产。主图现在使用同一 cold-process 计时范围，官方 hot-session 数字保留在收据中但不混入该图。
- 当前完整 SS/SLat draw 序列的 CUDA sampler 与 PyTorch 逐字节一致。Vulkan portable sampler 比较 243,029 个 F32 值，max abs 为 `2.205371856689453e-6`、mean abs 为 `1.6747961626876046e-7`，低于已有 `3e-6` / `4e-7` 限制。这不是跨 libm 的逐字节等价。
- PyTorch CUDA `randperm` 的 42,000 和 100,000 元素案例，以及含负坐标的完整去重/下采样规则有独立回归。CPU/Vulkan 生产路径使用显式 Philox 契约，不再回退 `mt19937`。
- 受控后处理的 positions、indices、纹理和法线都有独立测量。审查发现旧 UV 检查只对比烘焙前张量，遗漏了 GLB 导出转换；不能继续把旧 `uv_bitwise_equal` 称为官方最终 GLB 的 UV 一致。修复后报告直接比较官方 GLB，详见 R02。
- 纹理学习率已按官方“当前 Adam 更新后设置下一次学习率”排序；法线使用 F64 角度加权。其独立测试不能替代最终纹理和 GLB 测量。
- GGML 修改仍只有 `third_party/ggml-patches/0001-sam3d-ggml-combined.patch`。CMake 自动应用；helper 的 `--check` 已修复为只读，并拒绝额外未记录修改。干净克隆重放、二次幂等和最终 Git tree 相等已测试。
- `verify_native_runtime.py` 同时拒绝 `libcudnn`、`libtorch`、`libpython` 和缺失共享库。此检查覆盖链接依赖树，不冒充任意未来插件 `dlopen` 的追踪。
- 发布器验证 GLB 内嵌 PNG 与独立 PNG 像素完全相同。官方 GLB 可以合法省略 NORMAL；其缺失不能被错误当成无效 GLB。
- 模型准备补齐独立 MoGe 下载/转换；生成模型下载器校验实际发布大小和 GGUF 头，拒绝默认覆盖不同版本。公开权重可能早于当前转换器，不能仅凭同名文件声称和本机基线相同。
- 根目录 Python 启动器实跑曾在 FlexiCubes 分配 512 MiB 时 OOM。源码确认 SLat 采样结束后，生成器与条件编码器仍驻留 GPU，直至网格解码后才移走。现提前移走这两个不参与解码的模块；同图同 seed 重跑成功，Torch allocated 从 `4,503,961,088` 降到 `2,050,981,888` 字节，释放 `2,452,979,200` 字节（约 2.28 GiB）。没有改变模型 dtype、采样或烘焙规格。原失败日志保留，官方完整计时单独刷新，避免使用修复前时间冒充修复后性能。

## 3. 确认仍存在的问题

### R01：raw 数值误差不是 Q4 独有

修复导出之前，CUDA F16 的最终 GLB RGB MAE 为 `0.0802606350`、轮廓 IoU 为 `0.9771207237`；Vulkan F16 分别为 `0.0825248182`、`0.9766804388`。其中存在 R02 的非量化导出错误，不能把这些误差全部归给模型或后端计算。修复后的六组完整指标及实际纹理外观以当前报告为准；Gaussian 原始渲染误差与导出错误分开记录。

修复顺序：固定实际参考 dtype、TF32、SDPA、CFG/Euler 和布局；使用现有 `run_operator_oracle.py` 的 `same-gguf-dequantized` 范围先隔离原生图误差，再与 official checkpoint 比较权重误差。跨步模式覆盖 canonical 25 步条件/无条件 velocity、latent 和终态 support。不能用官方 support 或初始 latent 替换 raw 输入后宣称端到端对齐。

本轮额外把官方完整计时输出与官方阶段捕获输出直接比较；使用显存生命周期修复后重新生成的官方 GLB，60 视图 RGB MAE 为 `0.005726322050516804`、IoU 为 `0.9991253882268425`，法线平均角差 `2.7828572909037272` 度、最大角差 `168.68345642089844` 度。二者本身并非逐字节一致；这是两个官方入口/运行的实测差异，不是已定位到某个算子的根因，也不足以单独校准稳定阈值。其报告保存在当前归档的 `official_reference_agreement/`。修复后原生 F16/Q8 最终 GLB RGB MAE 约 `0.0257–0.0281`，明显高于这次官方对照差异；Q4 约 `0.0480–0.0511`，不能宣称已完全对齐。

### R02：GLB 导出 UV 与材质边界遗漏（已修复）

实际受控 GLB 对比：顶点和面完全一致，但 UV 最大绝对差为 `0.9991327524185181`；仅把原生 V 改为 `1 - V` 后最大差恰好为 **0**。本机 trimesh 的 `exchange/gltf.py` 在导出时明确执行该变换，原生此前直接写入 xatlas UV。旧检查比较的是 `asset_uv.samt`，位于官方导出变换之前，因而错误地认可了此边界。该诊断不需要推测量化误差。

同时，官方 `PBRMaterial` 未写 metallicFactor，glTF 有效默认值为 **1.0**；原生此前显式输出 **0.0**。二者 roughnessFactor 都是 1.0。当前共用 Lambertian renderer 不使用金属度，因此不能把其 RGB MAE 归因于 metallicFactor。

实现已在 `native_pbr_pipeline.cpp` 烘焙完成后翻转 V，并显式设置官方 baseColor、metallic 和 roughness；不改烘焙输入，也不改变通用资产 API 的默认值。C++ 测试检查烘焙与导出 UV 关系，Python 受控检查直接读取最终官方 GLB 的 UV 和材质。旧输出由新检查明确拒绝，修复后的主矩阵重新从 raw 图像运行，不通过修改既有 GLB 文件伪造修复后证据。物理 PBR 着色仍需单独 renderer 验证；材质字段已成为独立 coverage。

以同一修复前官方中间输入重新执行原生后处理，导出 UV 已逐字节对齐。60 视图、512px 的受控最终 GLB 实测 RGB MAE `0.0011920422635739668`、IoU `1.0`、深度 MAE `0.0`。这是导出修复的独立事实，不是六组 raw 模型精度全部通过的证据；小型诊断收据保存在当前归档的 `audit/` 中。

官方这里输出烘焙 base-color 材质，并不预测独立 metallic、roughness、normal 纹理；“PBR 支持”不能扩张成官方未提供的材质预测能力。

### R03：完整接受条件仍不充分

- 原有 neural render `MAE <= 0.01` 保留，不能根据本轮误差放宽。
- 最终 GLB RGB、轮廓、法线、深度和受控 texture 的质量预算尚未由官方重复运行和明确质量要求校准；未配置阈值仅表示完成测量，不可 PASS。
- raw pose 的差异由当前发布器独立记录；decoder pose oracle 通过不等于整个网络预测 pose 通过。当前 GLB 为 decoder-local 坐标，官方同样将 pose 单独输出；完整场景放置应另验 pose。
- native 同契约 hot-session 基准尚未实现。现已完成的是可比较的完整 cold-process 测量，不应再声称“没有完整 GLB 计时”，也不能把单次 cold 结果称为稳定 p95。
- 本轮每行一次完整重建，不能证明稳定 70 秒、连续请求资源复用或跨场景泛化。测试还应覆盖多图像、多 support 规模和重复请求；本轮未重新完成 CPU 全模型 raw GLB 矩阵。
- 部署脚本在当前已配置环境测试，不等于在干净机器重新安装验证。部分 Python 依赖尚未完整锁版本；本机官方 Python 使用 VTK 9.6.2、OpenCV 5.0.0、trimesh 5.0.0，原生使用 VTK 9.3、OpenCV 4.5.4。当前受控结果是这组具体依赖的事实，不是任意版本组合的保证。
- 本机 `pip check` 实际失败：除刻意不安装的官方训练依赖外，OpenCV 5 / plyfile 1.1.5 的 NumPy >= 2 声明与当前 NumPy 1.26.4 冲突。安装脚本已把 NumPy、OpenCV 4.9.0.80、plyfile 1.0.3 放在同一解析事务，并锁定几何处理关键包；未在计时期间修改当前环境。修正安装组合的全新部署及 oracle 复测仍待完成，不能将本轮结果冒充该新环境的数据。

### R04：量化改进的入口

先解决 F16/参考实现差异，再用长窗口最终 latent、occupancy 和离散 support 门禁筛选敏感层保留；只有候选通过 canonical `0 -> 25` 门禁才投入完整 raw E2E。必要时使用跨步、多敏感层 QAT 或显式 `Wq@x + B@(A@x)` 残差表示。当前 same-GGUF oracle 拒绝尚未建模的残差 GGUF，不能直接折叠权重掩盖运算顺序差异。

保留的 Q4 最佳实验是独立的 SS Q4_K 敏感层模型，本轮六行矩阵的 Q4 是统一 Q4_0。二者不互相代替；没有新增训练结果或通过承诺。

### R05：性能与工作量

完整 cold GLB 包含网格、UV、观测和烘焙，不能套用先前仅神经推理“接近 70 秒”的结论。本轮按要求收尾，没有继续优化。也没有足够的阶段 GPU profile 支持“因为没有 cuDNN 才慢”的归因。

2026-09-10 的修复后同口径测量：官方在显存生命周期修复后独占计算 GPU 重测为 `234.303 s`，六组原生 `215.338–252.136 s`。Q8/Q4 快于该单次官方 cold-process，但 CUDA F16 的 `252.136 s` 和 Vulkan F16 接 CUDA PBR 的 `236.220 s` 仍较慢。所有原生行均超过 70 秒，Vulkan Q4 的 `224.024 s` 也未快于 Q8 的 `223.370 s`。原生二进制未改变，保留原六行 raw 实测；官方原始 `264.464 s` 收据作为历史来源保留，不再用于当前图表。单次小差异不代表稳定排序；完整数字、样本数、刷新时间和脚本哈希见当前报告。

未来重新授权性能工作时，优先测量 strict attention 的 score 物化、decoder 窗口 mask、实际 M/N/K 与累加 dtype、重复模型加载、跨阶段搬运及 Vulkan submit/fence 等待。分块 attention 主要减少中间 IO，真正窗口计算才将工作量从全局平方改为窗口平方和。不能凭源码候选顺序宣称某项是已测得的最大瓶颈；不得无条件删除同步或增加超出显存预算的缓存。

## 4. 发布与复测

完整可执行命令集中在 [benchmarks/README.md](../benchmarks/README.md)，部署/许可证/下载集中在 [运行指南](../README.md)，验收定义在 [Native E2E Parity Contract](NATIVE_E2E_PARITY_CONTRACT.md)。失败返回码是测量结论，不应通过 `|| true` 当成验收成功。

本轮主图只使用完整 GLB cold-process 数据，保留原始汇总与依赖收据。旧 PLY 展示图、旧 normal 资产和被替代的大型 raw dump 清理；GGUF 均留在 `models/gguf/`，仍被测试引用的固定 stage 输入不删除。归档不包含所有大型中间张量，原始收据中的临时目录是来源记录，不是可再次打开的部署路径；可用 GLB 和 PNG 由当前矩阵的相对路径与 SHA-256 定位。

后续验收必须同时满足 raw 精度、材质、明确数值预算、同口径速度、重复运行稳定性、全部后端行以及单 patch 可复现，不能用某一个通过项替代其余缺项。
