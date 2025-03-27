# RD-AD2VLA
Pre-research on VLA. R(einforcement learning)D(ual stream)-AD's "closed-loop evaluation" and "reinforcement learning" can be reused.
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15073856.svg)](https://doi.org/10.5281/zenodo.15073856)


# Status
Early-stage code framework. Core classes and logic are being implemented.  


# Overall Architecture

**※ Flowchart:**

Multi-camera input → [3D Gaussian Scene Encoder] → 3D Semantic Gaussian Representation  
                      ↓  
[System 2: VLM Reasoning Module] → Task Semantic Features + Long-term Path Planning  
                      ⇵ (Intelligent Mode Switching)  
[Fast Thinking Mode] → Emergency Action Sequence  
                      ↓  
[System 1: DIT + Flow Matching Action Generation] → Vehicle Control Commands (steering angle, throttle, brake, etc.)  


# Module 1: 3D Gaussian Scene Representation Encoding (GaussianAD 3DGS)

**※ Input:**

1.Multi-camera RGB images (6-8 surround-view cameras, resolution 1920×1080@30FPS)

2.LiDAR point cloud (optional, for initializing Gaussian centers)

3.Navigation map data (vectorized lane topology, speed limits, traffic light coordinates, etc.)

**※ Processing Pipeline:**

1.Gaussian Center Initialization:

1.1 Image features are extracted via a Transformer to generate initial 3D Gaussian centers (x,y,z coordinates).

1.2 LiDAR point clouds or monocular depth estimation refine positional accuracy (error <0.1m).

2.Attribute Encoding:

2.1 Each Gaussian center carries:

  --Geometric attributes: Covariance matrix (controls ellipsoid shape), opacity.
 
  --Semantic attributes: Class (vehicle/pedestrian/obstacle), motion states (velocity, acceleration).
 
  --Physical attributes: Surface material (affects braking distance calculation).
 
3.Dynamic Update Mechanism:

3.1 Spatiotemporal Transformer Motion Flow Prediction:

3.2 Motion correlations are modeled across time (5 historical frames) and space (adjacent Gaussian centers) via stacked spatiotemporal attention layers.

3.3 Outputs displacement vector fields (Δx,Δy,Δz) and velocity changes for each Gaussian center over a 2-second horizon.

**※ Output:**

1.3D Gaussian Scene Tensor (dimension: N×15, where N = number of Gaussian centers; 15 dimensions include coordinates, covariance, semantic labels, etc.).

2.Key Semantic Features:

  --Road structure: Lane curvature, drivable area masks.
 
  --Dynamic objects: Trajectory predictions for surrounding vehicles, pedestrian intent classification (probability distribution).
 
  --Environmental states: Road friction coefficient estimation, visibility score.
 

# Module 2: VLM Reasoning Module in Dual-System Architecture (GR00T N1 System 2)

**※ Input:**

1.3D Gaussian scene tensor from Module 1.

2.Navigation instructions (natural language, e.g., "Turn right into the ramp 300m ahead, maintain 60km/h").

3.Real-time risk scores (collision probabilities of dynamic objects from Module 1).

**※ Processing Pipeline:**

1.Dual-Mode Intelligence Switching:

1.1 Slow Thinking Mode (default):

  --Executes hierarchical reasoning (strategic → tactical → rule layers), processing time: 50-200ms.
 
  --Applicable to routine scenarios (car-following, lane changes, intersection navigation).
 
1.2 Fast Thinking Mode (emergency):

  --Auto-triggered when risk score >0.7 (e.g., sudden pedestrian intrusion).
 
  --Bypasses strategic layer to invoke pre-trained emergency policy library (e.g., AEB trigger logic).
 
  --Response latency <20ms, outputs coarse-grained obstacle avoidance paths.
 
2.Cross-Modal Alignment:

2.1 Projects 3D Gaussian tensor into VLM token space (512D) for cross-attention computation with language instructions.

3.Hierarchical Reasoning:

3.1 Strategic Layer: Generates global paths (e.g., highway ramp vs. service road), time horizon: 30-60s.

3.2 Tactical Layer: Plans local action sequences (e.g., overtaking, car-following, emergency avoidance), time horizon: 5-10s.

3.3 Rule Layer: Validates compliance with traffic regulations (e.g., lane change legality at dashed lines) in real time (March 27, 2025).

**※ Output:**

1.Dual-Mode Joint Output:

{
  "mode": "slow" | "fast",  # Current decision mode
  
  "trajectory": [...],       # Detailed path (slow) or avoidance path (fast)
  
  "confidence": 0.92,        # Decision confidence
  
  "emergency_override": True  # Whether active braking/steering is triggered
}

2.Interpretability Report (natural language, e.g., "Left truck blind spot detected; fast thinking mode activated").


# Module 3: DIT + Flow Matching Action Generation (GR00T N1 System 1)

**※ Input:**

1.Dual-mode output from Module 2.

2.Real-time (March 27, 2025) vehicle states: Steering angle, wheel speed, yaw rate (CAN bus data at 10ms intervals).

3.Physical constraints: Maximum steering angular rate, braking acceleration limits, etc.

**※ Processing Pipeline:**

1.Mode-Adaptive Flow Matching:

1.1 Slow mode: Constructs manifold from current state to strategic path endpoint (time horizon: 1-5s).

1.2 Fast mode: Constructs emergency obstacle avoidance manifold (time horizon: 0.2-0.5s; higher-order derivative constraints).

2.DiT Diffusion Process:

2.1 Initial noisy action sequence → Denoised progressively via 12-layer cross-attention DiT blocks.

3.Control Command Fusion:

3.1 Injects predefined safe control values (e.g., maximum braking force) when emergency_override=True.

**※ Output:**

1.Multimodal Control Commands (100Hz high-frequency output):

[
  {"t":0.00, "steer":0.12, "throttle":0.35, "brake":0.0, "mode":"slow"},
  
  {"t":0.01, "steer":0.28, "throttle":0.0, "brake":0.8, "mode":"fast"},
  
  ... # Dynamically blended control signals
]

2.Mode Switch Log: Records environmental snapshots and decision rationale for each fast-mode activation.


# Code Organization

RD-AD2VLA/

├── configs/

│ ├── gauss_encoder.yaml

│ └── vlm_agent.yaml

├── data/

├── models/

│ ├── scene_encoder/

│ ├── cot-VLM/

│ └── diffusion/

├── utils/

│ ├── data_augment.py

│ └── safety_check.py

└── scripts/

│ ├── train_gauss.py

│ └── train_vla.py


# Plan
Due to the fact that pre-research is updated in spare time, it will be updated module by module from Q1 to Q4 of 2025.


# License
[MIT License](LICENSE)  
