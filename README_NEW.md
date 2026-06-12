# 本地修改说明

本文档记录当前仓库相对原始 Genesis World 仓库做过的本地修改，主要集中在 `examples/trampoline/` 里的 G1 蹦床站稳任务。

## G1 蹦床站稳任务

本次修改的目标不是让机器人主动跳高，而是让 Unitree G1 在 PBD cloth 蹦床上尽量稳定站立。任务仍然保留蹦床的柔性接触和机器人-蹦床耦合，但 MDP、控制频率、动作空间、观测和奖励都做了调整。

## MDP 与观测

- 将 `G1TrampolineEnv` 从只控制 12 个腿部关节，改为控制 G1 的 29 个可驱动关节。
- 上身不再被硬锁住，腰部、手臂和手腕都可以被策略观测和控制。
- 任务目标保持为站稳，而不是跳跃。
- 移除了原来几乎恒定的 phase/command 观测。
- 新增脚下局部蹦床布料观测，而不是直接输入全部 cloth particles。

当前策略观测包含：

- base 角速度
- 重力方向在机体坐标系下的投影
- base 相对初始站立高度
- base 垂直速度
- base 水平位置
- 左右脚附近的蹦床局部高度和垂向速度摘要
- 全身关节相对默认角度
- 全身关节速度
- 上一时刻动作

这样既能让策略感知脚下蹦床压缩和回弹状态，又不会把所有 cloth 粒子都塞进观测导致维度过高。

## 控制频率

- 物理仿真步长保持为 `dt = 0.005` 秒，即 200Hz。
- 新增 `control_decimation = 4`。
- 一个 RL/control step 会执行 4 个物理步。
- 因此策略控制频率是 50Hz。
- 8 秒 episode 对应约 400 个 control step，而不是之前约 1600 个 physics step。
- 为兼容旧 checkpoint，旧配置中如果没有 `control_decimation`，会回退到 `1`。

## 动作空间、Action Scale 与 PD 参数

- 移除了所有关节共用一个 `action_scale = 0.45` 的做法。
- 新增 per-joint action scale。
- action scale 会根据 MJCF 里的 joint limit、默认关节角和安全系数计算，再经过任务级上限截断。
- 这样可以避免 hip/yaw 这类大范围关节动作过大，也避免 ankle roll 这类小范围关节频繁撞限位。

PD 参数也改成按 G1 关节组设置：

- hip / knee / waist / shoulder / elbow 使用较高位置增益。
- ankle / wrist 使用较低位置增益。

这个设置参考了 G1 MJX 资产里的关节分组思路，比原来所有关节统一高增益更合理。

## 奖励设置

当前 reward 仍然是站稳任务奖励，主要鼓励：

- 存活
- 站立高度稳定
- 保持在蹦床中心
- 身体保持直立
- 减少水平漂移
- 减少垂直方向弹跳
- 减少 roll/pitch 角速度
- 动作平滑
- 关节速度和关节加速度较小
- 姿态不要过度偏离默认站姿

因为现在 29 个关节都可控，所以正则项做了分组权重：

- 下肢关节权重较低，允许主要通过腿部和脚踝调节平衡。
- 腰部权重中等，允许少量辅助调姿。
- 手臂和手腕权重更高，鼓励上身少动。

这样策略不是硬锁上身，而是通过 reward 自然倾向于“下肢为主，上身少动”。

## 蹦床接触和布料观测

- 保留 rigid-PBD coupling，让 G1 脚部和 PBD cloth 蹦床发生耦合。
- 保留临时 MJCF 脚部接触 patch 生成逻辑，用于增大脚和布料的有效接触面积。
- 新增左右脚附近 cloth 粒子的局部加权摘要：
  - 左脚相对 cloth 高度
  - 右脚相对 cloth 高度
  - 左脚附近 cloth 垂向速度
  - 右脚附近 cloth 垂向速度

这比完整 height grid 或全部粒子观测更轻量，也更贴近站稳控制需要的信号。

## 训练与评估

`examples/trampoline/trampoline_train.py` 使用 TensorBoard logger。

默认训练规模恢复为较保守的设置：

- `num_envs = 128`
- `max_iterations = 1200`

建议先 smoke test：

```bash
uv run python examples/trampoline/trampoline_train.py -B 16 --max_iterations 10
```

如果显存允许，再逐步增加并行环境数量：

```bash
uv run python examples/trampoline/trampoline_train.py -B 64 --max_iterations 1200
uv run python examples/trampoline/trampoline_train.py -B 128 --max_iterations 1200
```

评估命令示例：

```bash
uv run examples/trampoline/trampoline_eval.py --ckpt 2500
```

注意：如果 checkpoint 是用旧版 obs/action 维度训练出来的，可能无法直接兼容当前环境。

## 依赖修改

在 `pyproject.toml` 中加入了 RL 训练需要的依赖：

- `rsl-rl-lib>=5.0.0`
- `tensorboard`

同时移除了 `dev` extra 中直接声明的：

- `matplotlib>=3.7.0`

`uv.lock` 中仍可能因为 `vtk`、`open3d` 或其他依赖链出现 `matplotlib`，但当前仓库不再通过 `dev` extra 主动要求安装它。
