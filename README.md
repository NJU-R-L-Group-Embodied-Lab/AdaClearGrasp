# AdaClearGrasp: Learning Adaptive Clearing for Zero-Shot Robust Dexterous Grasping in Densely Cluttered Environments

**AdaClearGrasp** is a hierarchical robot manipulation framework designed to solve robust dexterous grasping tasks in cluttered environments. It leverages the high-level semantic reasoning capabilities of Vision-Language Models (VLMs) combined with a library of robust low-level atomic skills to perform adaptive obstacle clearing and target grasping.

The core philosophy is **"High-Level Semantic Planning → Atomic Skill Scheduling → Low-Level Motion Control"**. By introducing the **Model Context Protocol (MCP)**, the robot's physical capabilities are encapsulated as standardized tools, allowing VLMs to control the robot directly through function calls.

## ✨ Key Features

*   **VLM-Guided Planning**: Utilizes advanced VLMs (e.g., Qwen-VL, GPT-4o) to perceive scene geometry and generate multi-step manipulation plans (e.g., "Push the can to the left, then grasp the apple").
*   **MCP Architecture**: Implements a **Model Context Protocol** server (`exec/mcp_server.py`) that exposes robotic skills as standardized tool interfaces.
*   **Hybrid Control Strategy**: Combines deterministic Inverse Kinematics (IK) for precise reaching and PPO-based Reinforcement Learning policies for robust grasping.
*   **Atomic Skill Library**: Provides a suite of robust motion primitives, including `move_to`, `push`, `pull`, `lift`, `lower`, and `grasp`.
*   **ManiSkill Integration**: Features a high-fidelity `PickClutterYCB-XArm7-v1` environment built on ManiSkill and Sapien.

---

## 🛠️ Installation

### 1. Prerequisites
*   Linux (Ubuntu 20.04/22.04 recommended)
*   Anaconda or Miniconda
*   NVIDIA GPU (for simulation rendering and VLM inference)

### 2. Steps

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/zjwqsd/AdaClearGrasp.git
    cd AdaClearGrasp
    ```

2.  **Create Conda Environment**
    Create the environment using the provided `environment.yml`:
    ```bash
    conda env create -f environment.yml
    conda activate clear
    ```

3.  **Download Assets**
    AdaClearGrasp relies on ManiSkill's YCB dataset:
    ```bash
    python -m mani_skill.utils.download_asset "PickSingleYCB-v1"
    ```

---

## ⚙️ Configuration

Before running, you need to configure runtime parameters (e.g., VLM API Key).

1.  **Create Configuration File**
    Copy the template file:
    ```bash
    cp configs/runtime_config_template.yaml configs/runtime_config.yaml
    ```

2.  **Edit Configuration**
    Open `configs/runtime_config.yaml` and fill in your API information. The default supports OpenAI-compatible interfaces (e.g., SiliconFlow):
    ```yaml
    openai:
      api_key: "sk-xxxxxxxxxxxxxxxx"
      base_url: "https://api.siliconflow.cn/v1"  # Or other providers
      model: "Qwen/Qwen3-VL-32B-Instruct"       # Or gpt-4o
    ```

---

## 🚀 Usage Guide

All commands should be run from the project root directory.

### 1. Test Atomic Skills
Before running complex VLM planning, it is recommended to test if the low-level atomic skills are working correctly. This script executes a sequence of predefined actions (reset, move, push/pull, grasp).

```bash
python scripts/test_skills.py
```
*Note: If running on a headless server, set `RUN_MODE = "video"` in the script to save screen recordings.*

### 2. Run VLM Planning (Autonomous Agent)
Launch the autonomous agent to let the VLM observe the scene and make decisions.

```bash
python -m plan.vlm_plan \
    --scene_name apple \
    --clutter_count 4 \
    --scene_id 1 \
    --max_plan_steps 40
```

**Parameters**:
*   `--scene_name`: Target object category (e.g., `apple`, `can`, `mug`).
*   `--clutter_count`: Number of clutter objects in the scene (e.g., `2`, `4`, `6`).
*   `--scene_id`: Specific scene ID (corresponding to config files in `data/scenes/`).

**Workflow**:
1.  Initialize the simulation environment.
2.  Render the RGB image of the current view.
3.  Send the image and task description to the VLM.
4.  VLM returns action instructions in JSON format (e.g., `{"action": "push", "args": {"side": "left", "dist_m": 0.1}}`).
5.  MCP Runtime parses and executes the action.
6.  Repeat until the task is completed or maximum steps are reached.

---

## 📂 Project Structure

```text
AdaClearGrasp/
├── configs/               # Configuration files
│   ├── runtime_config.yaml          # [User Created] Runtime config (API Key)
│   └── runtime_config_template.yaml # Config template
├── core/                  # Core algorithms
│   ├── kinematics/        # Robot Inverse Kinematics (IK)
│   └── perception/        # Point cloud processing & geometry
├── env/                   # Environment layer
│   └── sim/               # ManiSkill simulation (PickClutterYCB)
├── exec/                  # Execution layer (MCP Server & Skills)
│   ├── skills/            # Atomic skills implementation (Clear, Grasp, Move)
│   └── mcp_server.py      # MCP Server: Exposing skills to VLM
├── plan/                  # Planning layer (VLM Agent)
│   ├── vlm_plan.py        # Main planning loop
│   ├── mcp_runtime.py     # MCP client runtime
│   └── prompts.py         # VLM system prompts
├── scripts/               # Tool scripts
│   └── test_skills.py     # Skill testing script
├── data/                  # Data & Models
│   ├── models/ppo/        # Pre-trained Grasp RL policy
│   └── scenes/            # Scene definition JSON files
└── README.md
```

## 📝 License

This project is licensed under the MIT License.
