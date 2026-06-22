# steer

模型行为引导调控（LLM Behavior Steering）。

本项目以 **Qwen3-4B-Instruct-2507** 为基座模型，用于探索激活引导（activation steering）等大模型行为调控方法。

---

## 1. 环境要求

- macOS / Linux（macOS 推荐 Apple Silicon，使用 MPS 加速）
- [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) / Anaconda
- 磁盘空间：模型权重约 **8 GB**

---

## 2. 安装 Miniconda（已安装可跳过）

**macOS (Apple Silicon)：**

```bash
mkdir -p ~/miniconda3
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init zsh   # bash 用户改成 conda init bash
```

**macOS (Intel)：** 把上面 URL 中的 `arm64` 换成 `x86_64`。

**Linux (x86_64)：**

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init bash
```

重启终端使 `conda` 命令生效。

---

## 3. 创建虚拟环境并安装依赖

```bash
# 克隆/进入项目目录后
conda create -n steer python=3.11 -y
conda activate steer

# 安装项目依赖
pip install -r requirements.txt
```

> 如果在国内访问 PyPI 较慢，可加 `-i https://pypi.tuna.tsinghua.edu.cn/simple` 使用清华镜像。

---

## 4. 下载模型与数据

### 4.1 下载 Qwen3-4B 模型

```bash
python download-models.py
```

模型将下载到 `./qwen3-4b/`。约 8 GB，请保持网络畅通。

> 国内用户如无法访问 huggingface.co，可在执行前设置镜像：
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> ```

### 4.2 下载 SteerEval 评测数据（可选）

```bash
python download-data.py
```

数据将下载到 `./data/SteerEval/personality/`。

---

## 5. 运行推理示例

```bash
conda activate steer
python run.py
```
