# DOBOT 越疆视觉手眼标定系统（二次开发）

这是一个仿照演示软件制作的机械臂控制端。当前版本先完成了“机械臂控制部分”，
3D 视觉/标定部分界面已预留，逻辑待下一步实现。

## 运行

在 `robotgame` 环境里启动：

```powershell
E:\robot_software\envs\robotgame\python.exe D:\robot_projects\robotgame\dobot_handeye\main.py
```

或先激活环境：

```powershell
conda activate robotgame
cd D:\robot_projects\robotgame\dobot_handeye
python main.py
```

## 已实现功能

- 连接 / 断开机器人（Dashboard 端口 29999）
- 切换 TCP 控制模式（`RequestControl`）
- 上电、使能 / 下使能、清除报警、急停
- 开始 / 停止拖拽
- 设置全局速度、User/Tool 坐标系
- X/Y/Z/Rx/Ry/Rz 笛卡尔点动（按住移动，松开停止）
- 获取当前位姿、直线运动到目标点
- I/O 实时监控：16 路 DI 状态灯 + 16 路 DO 拨码
- 30004 实时反馈解析：位姿、机器人状态、T/U、DI/DO
- D435 相机实时预览
- 模型训练：一键划分数据集、训练 YOLO、生成 `best.pt` 并实时显示进度
- 实时日志与底部状态栏

## 待实现（3D 视觉部分）

- 标定流（手动捕获点位、自动球面采点、矩阵解算等）
- 位姿记录表格的采集与误差计算

## 模型训练流程

1. 用 X-AnyLabeling 完成标注，`Export -> YOLO HBB` 导出。
2. 把导出的图片和同名 `.txt` 放进 `dobot_handeye\dataset\raw\`。
3. 编辑 `dobot_handeye\dataset\class.txt`，每行一个类别名，顺序和标注时的 `classes.txt` 一致。
4. 在软件里点“模型训练 -> 训练”，训练日志会实时刷到界面右下角。
5. 训练完成后，最终模型保存在 `D:\robot_projects\robotgame\models\best.pt`。

更详细说明见 `dataset/README.md`。

## 协议要点

- 控制指令端口：`29999`，ASCII 文本，应答以 `;` 结尾
- 实时反馈端口：`30004`，1440 字节小端二进制，约 8ms 一包
- 常见指令：`PowerOn()`、`EnableRobot()`、`DisableRobot()`、
  `ClearError()`、`EmergencyStop(1)`、`StartDrag()`、`StopDrag()`、
  `SpeedFactor(80)`、`User(0)`、`Tool(0)`、`GetPose()`、
  `MoveJog(X+,coordtype=1,user=0)`、`MoveJog()`（停止）、
  `MovL(pose={x,y,z,rx,ry,rz},user=0,tool=0)`、
  `DOInstant(1,1)`、`DI(1)` 等。

## 安全提示

- 真机联调前，请确认急停回路可用、机械臂周围无人。
- “运动到点”默认使用直线运动 `MovL`，若目标点附近存在奇异位姿会报错，
  可将 `main_window.py` 中的 `MovL` 改成 `MovJ` 改用关节运动。
- 首次连接后建议先用低速（全局速度 10%~20%）验证点动方向。
