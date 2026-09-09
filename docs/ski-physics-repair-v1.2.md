# 环境接触修复 V1.2

本版本修复结构与接触层，并通过规定范围内的 CPU 夹具复测；不宣称完整 M0、闭环技能或正式 RL 已验收。

## 已落地的修复

- G1 碰撞掩码改为 `contype=4, conaffinity=7`，恢复自身碰撞；官方源文件、网格、质量、惯量及执行器参数不变。
- 雪板与雪的窄相位改为裸露高度场顶面三角形上的球面采样接触；保留原板体凸体／板刃体积代理处理板间及身体碰撞。MuJoCo 仍求解全部法向与各向异性摩擦力，没有额外根节点力或重复法向弹簧。
- 每段每条表面轨迹采样 32 站，只将局部距离极小值组成接触流形；采样密度增加不会直接等量增加约束。上下表面均支持碰撞，倒置时不激活刻滑抓刃。
- 刃线方向使用接触站的局部曲线切向，避免整段翘头弦向代替承载点方向。
- 物理步长 0.25 ms，法向接触响应时间从 12 ms 改为 6 ms；LLC 100 Hz、HLC 10 Hz。重新运行静载、滑行、冲击和换刃试验，没有通过放软接触掩盖穿透。

## 复测

- 基础资产检查 1373 项，修复探针 35 项，接触模式 13 项；失败数 0。
- 平坦高度场使用 0.0625／0.125／0.25 m 网格及横向相位偏移，对照无限平面。
- 2／4 m/s、35°立刃，比较 0.25／0.125 ms 和 32／64 站；分别检查速度、方向与位置差。
- 高地形保留完整无控制滚动试验，并另测从首次真实接触开始的 80 ms 冲量窗口。

| 场地 | 0.25 ms 最深穿透 | 0.125 ms 最深穿透 | 3 秒／2 秒末速度相对差 |
|---|---:|---:|---:|
| moguls_regular | 9.26 mm | 5.03 mm | 9.81% |
| moguls_seeded | 7.80 mm | 15.75 mm | 4.27% |
| tabletop_small | 4.21 mm | 4.94 mm | 0.11% |
| tabletop_steep | 8.45 mm | 9.99 mm | 0.17% |

## 验收边界与保留问题

**regular moguls 的无控制长时间翻滚末速度仍有 9.81% 差异，未达到原来的 5% 整段阈值。** 新增的首次冲击冲量检查不能替代这一失败；该项保留为长轨迹敏感性，完整 M0 仍未签收。不得据此启动未经分阶段验收的高坡技能训练。

100 N 世界横向外载并不等于每个刃点沿其局部横向承载。6 ms 版本中 35°夹具在 25 N 下保持、100／150 N 下失抓；测试明确记录有限摩擦锥和分布接触导致的这个结果，没有把 `mu*N` 当作保证的世界横向静摩擦容量。

只有 2 m/s 保守夹具支持低侧滑转弯结论；4 m/s 探针允许明显侧滑。固定 roll/pitch 的无功夹具不代表机器人自行完成动作。

雪面查询与碰撞采用同一高度场三角形。采样球的最近点搜索覆盖相邻顶面三角形；超过有效场地边界应由任务终止。CPU Python 参考实现不等于已优化的 MJX／GPU 批量训练后端。

## 文件与复现

- 配置：`assets/env/configs/design.json`。
- 核验：`assets/env/reports/repairs_v1_2/verification.json`、`contact_modes.json`、`verdict.json`。
- 原始失败及输入：`assets/env/reports/repair_baseline/`。

```bash
.venv/bin/python assets/env/tools/build_assets.py
.venv/bin/python assets/env/tools/attach_g1.py
.venv/bin/python assets/env/tools/validate_assets.py
.venv/bin/python assets/env/tools/verify_repairs.py
.venv/bin/python assets/env/tools/verify_contact_modes.py
.venv/bin/python assets/env/tools/write_repair_report.py
.venv/bin/python assets/env/tools/finalize_manifest.py
```
