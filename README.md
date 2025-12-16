# Qwen Plays Breakout: Text-Based PPO

A Reinforcement Learning framework tailored for training Large Language Models (specifically **Qwen**) to play **Atari Breakout** using purely text-based observations. This project leverages **Proximal Policy Optimization (PPO)** with Low-Rank Adaptation (LoRA) to fine-tune the model efficiently.

## 🚀 Key Features
* **Text-Based Wrapper**: Converts visual Atari frames into structured text descriptions (e.g., "Ball X: 99, Paddle X: 72, Moving DOWN").
* **PPO Implementation**: Custom PPO agent designed for Causal LM value heads.
* **Efficient Training**: Uses **LoRA** to train only a small subset of parameters.
* **Reward Shaping**: Custom reward mechanisms to guide the LLM towards gameplay fundamentals.
* **Video Recording**: Automatically records gameplay videos during training.

## 🛠️ Installation
1. **Clone the repository**:
    ```bash
    git clone https://github.com/oberpierre/hs25-rl-ppo.git
    cd hs25-rl-ppo
    ```
2. **Install Dependencies**:
    It is recommended to use a virtual environment.
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

## 🏃 Usage
### Training
Start a new training session with default hyperparameters (Qwen/Qwen3-0.6B):
```bash
python train.py
```
Outputs (logs, checkpoints, and videos) are saved in the `runs/` and `videos/` directories.

### Playing (Inference)
Watch the trained model play:
```bash
# Play 1 episode
python play.py --checkpoint checkpoints/[run_name]/step_[N]

# Play 5 episodes and get stats
python play.py --checkpoint checkpoints/[run_name]/step_[N] --episodes 5
```

### Grid Search
Run a hyperparameter sweep over Learning Rates and KL Targets:
```bash
python grid_search.py
```

### Verification
Run a quick environment check to ensure text observations and actions are working:
```bash
python verify_env.py
```

## 📂 Project Structure
- `train.py`: Entry point for training.
- `play.py`: Inference script for playing/evaluating checkpoints.
- `src/`
    - `env_wrapper.py`: Gymnasium wrapper converting RAM states to text.
    - `model.py`: ActorCritic architecture wrapping the HuggingFace model.
    - `ppo.py`: PPO algorithm implementation.
    - `reward_shaping.py`: Reward shaping logic (e.g., paddle-ball alignment).
- `checkpoints/`: Saved models.
- `runs/`: Tensorboard logs.
