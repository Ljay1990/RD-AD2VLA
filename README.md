# RD-AD2VLA
Pre-research on VLA. R(einforcement learning)D(ual stream)-AD's "closed-loop evaluation" and "reinforcement learning" can be reused.
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15073856.svg)](https://doi.org/10.5281/zenodo.15073856)

# Status
Early-stage code framework. Core classes and logic are being implemented.  

# Plan
Due to the fact that pre research is updated in spare time, it will be updated module by module from Q1 to Q4 of 2025.

# Overall
在之前“端到端+VLM”自动驾驶架构的数据准备、模型升级、模仿强化、闭环评测、压缩部署等匆忙预研与量产准备的基础上，迈向VLA自动驾驶架构的预研工作，聚焦于模型架构设计：以 3DGS 场景表征为信息中枢、通过 Action Token 衔接 CoT-VLA 与 Diffusion 模块。

**一、3DGS 场景表征模块的架构设计：**

1.3D 高斯初始化与优化

几何初始化：在自车坐标系内均匀分布N个3D 高斯点，初始参数包含位置μ0、各向异性协方差Σ0、语义概率s0

动态参数预测：通过时空 Transformer 预测每个高斯的运动流 Δμt，建模障碍物运动趋势

2.稀疏特征压缩

重要性剪枝：计算高斯点的语义显著度 S=max (s)⋅ Δμ，保留M个关键点

特征降维：使用 MinkowskiNet 的稀疏 3D 卷积层，将 256 维高斯特征压缩至 512 维全局场景向量 fscene

**二、CoT-VLA 推理决策模块的架构设计：**

1.多模态对齐层

语言编码：Whisper-large-v3 提取语音文本，ALBERT 生成指令嵌入 flang

跨模态注意力：Attention(Q=fscene, K=V=flang)

2.双模推理引擎

慢思考模式：基于逻辑模板的显式推理（如解析"右转避让"指令并计算安全转向角）

快思考模式：6 层 Transformer Decoder 实现隐式推理，动态路由阈值 τ

**三、Diffusion 轨迹优化模块的架构设计：**

1.锚定扩散策略

多模态锚点先验：通过 K-means 聚类预训练获取 K个典型驾驶模式锚点ak

截断扩散过程：仅需 2 步去噪（原需 20 步），时间步从 t 开始反向过程

2.级联扩散解码器

注意力机制重构：轨迹坐标与 Action Token 通过交叉注意力交互

轨迹解耦头：自车与他车轨迹联合优化

# Model Code
class RD-AD2VLA(nn.Module):

 def __init__(self):
 
 super().__init__()
 
 self.scene_encoder = SceneEncoder3DVLA() # 3D 场景
 
 self.decision_maker = OpenVLADecisionModule() # 推理决策
 
 self.trajectory_generator = AnchorDiffusionDrive() # 轨迹优化（锚定扩散策略）
 
 def forward(self, sensor_data, lang_inst):
 
 scene_feat = self.scene_encoder(sensor_data) # 传感器数据包含环视图像+LiDAR 点云
 
 action_token = self.decision_maker(scene_feat, lang_inst) # 语言指令编码与多模态对齐
 
 trajectory = self.trajectory_generator(action_token) # 生成联合轨迹
 
 return trajectory

# Data Format

dataset_config = {

 "3D_scene": {
 
 "sensors": {
 
 "multi_cam": (6, 3, 256, 256), # 环视 6 相机图像
 
 "lidar": (100000, 4), # 点云(x,y,z,反射强度)
 
 "gaussian_params": { # 3D 高斯初始化参数
 
 "positions": (100000, 3),
 
 "covariance": (100000, 3, 3),
 
 "sem_probs": (100000, 20)
 
 }

 }
 
 },
 
 "language": {
 
 "instruction": "str", # 自然语言指令
 
 "speech_wav": (16000*5,) # 5 秒语音片段
 
 },
 
 "action": {
 
 "trajectory": (51, 3), # 5 秒轨迹（10Hz）
 
 "action_token": (64,) # 结构化动作编码
 
 }
 
}

# Train Pipeline

三阶段训练策略：

**阶段 1：多模态联合训练（冻结场景编码器）**

python train_VLA.py \
 --freeze_scene_encoder \
 --lang_aug mix_whisper_albert # 语音文本联合编码

**阶段 2：端到端微调（全参数更新）**

python finetune_full.py \
 --use_lora # 参数高效微调 \
 --quant 4bit # 4 比特量化部署优化

**阶段 3：基于3DGS环境的强化学习**

同“端到端+VLM”架构的RD-AD项目PPO实践~

# Code Organization

RD-AD2VLA/

├── models/

│ ├── scene_encoder/ # 3D 场景编码组件

│ ├── CoT-VLA/ # 多模态决策模块

│ └── diffusion/ # 轨迹生成器

├── data/

│ ├── hybrid_dataset.py # 混合数据加载器

│ └── transforms/ # 多模态数据增强

├── configs/ # 训练配置文件

├── scripts/ # 分布式训练启动脚本

└── utils/

│ ├── minkowski_ops.py # 稀疏卷积优化
 
│ └── fsdp_wrapper.py # PyTorch FSDP 封装

# License
[MIT License](LICENSE)  
