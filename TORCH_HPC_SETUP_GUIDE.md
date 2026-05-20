# OMatG Training Setup on HPC with Singularity

Complete guide for setting up OMatG training on HPC using Singularity containers, from initial setup to job monitoring.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Setting Up Singularity Container](#setting-up-singularity-container)
3. [Installing Python and Conda](#installing-python-and-conda)
4. [Installing PyTorch and Dependencies](#installing-pytorch-and-dependencies)
5. [Setting Up OMG Fork](#setting-up-omg-fork)
6. [Creating Environment Setup Script](#creating-environment-setup-script)
7. [Creating Training Job Script](#creating-training-job-script)
8. [Submitting and Monitoring Jobs](#submitting-and-monitoring-jobs)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- Access to HPC with Singularity installed
- GPU nodes with CUDA 12.6+ support
- SLURM job scheduler
- Your code repository accessible on HPC

---

## Setting Up Singularity Container

### 1. Choose the Right Container Image

Use **Ubuntu 22.04.5** with CUDA 12.6.3 (matches your Modal setup):

```bash
# Check available CUDA containers
ls /share/apps/images | grep cuda

# Use this image:
/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
```

**Why Ubuntu 22.04.5?**
- Matches CUDA 12.6.3 requirements
- Widely available on HPC systems
- Good compatibility with PyTorch 2.8.0

### 2. Create Overlay File (for persistent storage)

```bash
# Create overlay file (15GB recommended, adjust size as needed)
singularity overlay create --size 25000 overlay-15GB-500K.ext3

# Verify it was created
ls -lh overlay-15GB-500K.ext3
```

**Note:** The overlay file stores all your installed packages and persists between container runs.

### 3. Enter Container Interactively

```bash
singularity exec --nv \
    --overlay overlay-15GB-500K.ext3:rw \
    /share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
    /bin/bash
```

**Important:** The `--nv` flag enables GPU support (required for CUDA).

---

## Installing Python and Conda

### 1. Install Miniconda

```bash
# Inside the container
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /ext3/miniconda3
export PATH=/ext3/miniconda3/bin:$PATH

# Verify installation
conda --version
```

### 2. Create Conda Environment

```bash
# Create environment with Python 3.11
conda create -n omatg python=3.11 -y

# Activate environment
source /ext3/miniconda3/etc/profile.d/conda.sh
conda activate omatg

# Verify Python version
python --version  # Should show Python 3.11.x
```

---

## Installing PyTorch and Dependencies

### 1. Install PyTorch with CUDA 12.6

```bash
# Make sure you're in the conda environment
conda activate omatg

# Install PyTorch 2.8.0 with CUDA 12.6
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu126

# Verify CUDA is available
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Expected output:** `CUDA available: True` (if using `--nv` flag)

### 2. Install PyTorch Geometric Packages

```bash
# Install from pre-built wheels (recommended - avoids build issues)
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.8.0+cu126.html

# Install torch-geometric
pip install torch-geometric==2.7.0
```

**Why pre-built wheels?** Building from source requires compilation and can fail. Pre-built wheels are faster and more reliable.

### 3. Install Other Dependencies

```bash
pip install lightning==2.5.0 \
    wandb \
    ase \
    pymatgen \
    matminer \
    smact \
    spglib \
    lmdb \
    numpy \
    scipy \
    pandas \
    matplotlib \
    plotly \
    tqdm \
    loguru \
    'jsonargparse[signatures]' \
    torchdiffeq \
    torchmetrics \
    torchsde
```

---

## Setting Up OMG Fork

### 1. Clone or Copy Your Fork

```bash
# Exit container first (if inside)
exit

# On HPC, navigate to your workspace
cd /scratch/kd2862/omatg-ghosted/

# Clone your fork (or copy it from elsewhere)
git clone <your-omg-fork-repo-url> omg-fork
# OR if you already have it:
# cp -r /path/to/omg-fork /scratch/kd2862/omatg-ghosted/
```

### 2. Install OMG Package

```bash
# Enter container again
singularity exec --nv --overlay overlay-25GB-500K.ext3:rw \
    /share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
    /bin/bash

# Activate conda environment
source /ext3/miniconda3/etc/profile.d/conda.sh
conda activate omatg

# Navigate to omg-fork and install
cd /scratch/kd2862/omatg-ghosted/omg-fork
pip install -e .

# Verify installation
python -c "import omg; print('OMG installed successfully')"
```

---

## Creating Environment Setup Script

Create `/ext3/env.sh` to automatically activate your environment:

```bash
# Inside container
cat > /ext3/env.sh << 'EOF'
#!/bin/bash
source /ext3/miniconda3/etc/profile.d/conda.sh
conda activate omatg
EOF

# Make it executable
chmod +x /ext3/env.sh

# Test it
source /ext3/env.sh
python --version  # Should show Python 3.11.x
```

**Note:** Adjust path if you installed Miniconda to `/ext3/miniforge3` instead:
```bash
source /ext3/miniforge3/etc/profile.d/conda.sh
```

This script will be sourced in your job script to activate the environment automatically.

---

## Creating Training Job Script

Create `run_train.SBATCH`:

```bash
#!/bin/bash

#SBATCH --account=torch_pr_148_general
#SBATCH --nodes=1
#SBATCH --time=47:59:00
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --job-name=omg-ghost-training-kd
#SBATCH --output=omg-ghost-training-kd.out

# Initialize - change to your omg-fork directory
cd /scratch/kd2862/omatg-ghosted/omg-fork/

# Run training (NO srun needed - script already runs in SLURM allocation)
singularity exec --nv \
    --overlay /scratch/kd2862/omatg-ghosted/overlay-15GB-500K.ext3:rw \
    /share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
    /bin/bash -c "source /ext3/env.sh && omg fit --conf=omg/conf_examples/csp_ghosted_linode_mp_20_j05.yaml"
```

**Important points:**
- Use **full absolute path** to overlay file
- Use `--nv` flag for GPU support
- **NO `srun`** needed (script already runs in SLURM allocation)
- Use `&&` to chain commands in bash -c

---

## Submitting and Monitoring Jobs

### 1. Submit Job

```bash
sbatch run_train.SBATCH
```

**Output:** `Submitted batch job 3177496`

### 2. Check Job Status

```bash
# Check if job is queued/running
squeue -u kd2862

# Check specific job
squeue -j 3177496

# See detailed info
scontrol show job 3177496
```

**Job states:**
- `PD` = Pending (waiting in queue)
- `R` = Running
- `CG` = Completing
- `F` = Failed

### 3. Monitor Output

```bash
# View output file in real-time
tail -f omg-ghost-training-kd.out

# Or watch it periodically
watch -n 10 'tail -n 50 omg-ghost-training-kd.out'

# View full output
cat omg-ghost-training-kd.out
```

### 4. Check Job History

```bash
# See job details including exit code
sacct -j 3177496 --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS

# See all your recent jobs
sacct -u kd2862 --format=JobID,JobName,State,ExitCode,Elapsed
```

**Exit codes:**
- `0:0` = Success
- Non-zero = Failed

### 5. Cancel Job (if needed)

```bash
scancel 3177496
```

---

## Troubleshooting

### CUDA Not Available

**Problem:** `CUDA available: False`

**Solution:** Make sure you use `--nv` flag:
```bash
singularity exec --nv --overlay ...
```

### Job Fails Immediately

**Check:**
```bash
cat omg-ghost-training-kd.out
```

**Common issues:**
1. **Missing `env.sh`** → Create it (see [Creating Environment Setup Script](#creating-environment-setup-script))
2. **Wrong overlay path** → Use full absolute path
3. **Config file not found** → Verify path to YAML file
4. **Import errors** → Check if packages are installed

### CPU Binding Error

**Error:** `srun: error: CPU binding outside of job step allocation`

**Solution:** Remove `srun` from your script (see [Creating Training Job Script](#creating-training-job-script))

**Explanation:** Your batch script already runs in a SLURM allocation. Using `srun` tries to create a nested allocation, causing conflicts.

### PyTorch Geometric Build Fails

**Error:** `ModuleNotFoundError: No module named 'torch'` during build

**Solution:** Install from pre-built wheels:
```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.8.0+cu126.html
```

### Overlay File Full

**Check space:**
```bash
# Inside container
df -h /ext3
```

**Solution:** Create larger overlay or clean up:
```bash
singularity overlay create --size 50000 overlay-50GB-500K.ext3
```

---

## Quick Reference Commands

```bash
# Enter container interactively
singularity exec --nv --overlay overlay-25GB-500K.ext3:rw \
    /scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
    /bin/bash

# Activate environment (inside container)
source /ext3/env.sh

# Submit job
sbatch run_train.SBATCH

# Check job status
squeue -u kd2862

# Monitor output
tail -f omg-ghost-training-kd.out

# Check job history
sacct -j JOBID --format=JobID,JobName,State,ExitCode,Elapsed
```

---

## Summary Checklist

- [ ] Created overlay file
- [ ] Installed Miniconda in container
- [ ] Created conda environment with Python 3.11
- [ ] Installed PyTorch 2.8.0 with CUDA 12.6
- [ ] Installed PyTorch Geometric packages
- [ ] Installed other dependencies
- [ ] Cloned/copied omg-fork
- [ ] Installed omg package (`pip install -e .`)
- [ ] Created `/ext3/env.sh` script
- [ ] Created training job script (`run_train.SBATCH`)
- [ ] Verified config file exists
- [ ] Submitted job with `sbatch`
- [ ] Monitoring job output

---

**Note:** Adjust all paths (`/scratch/kd2862/omatg-ghosted/`, etc.) to match your HPC setup.

