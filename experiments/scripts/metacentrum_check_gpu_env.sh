#!/bin/bash
#PBS -N batl_check_gpu_env
#PBS -l select=1:ncpus=1:ngpus=1:gpu_mem=10gb:gpu_cap="cuda86|cuda90|cuda120":mem=8gb:scratch_local=10gb
#PBS -l walltime=00:15:00
#PBS -j oe
#PBS -o /auto/brno2/home/jozefsprlak/logs/batl/check_gpu_env.log
#PBS -m abe
#PBS -M 536343@mail.muni.cz

set -u

trap 'command -v clean_scratch >/dev/null 2>&1 && clean_scratch || true' EXIT
cd $SCRATCHDIR

echo "===== Node / GPU ====="
hostname
nvidia-smi || echo "WARNING: nvidia-smi unavailable"

echo
echo "===== Activating env ====="
ENV_PATH="${ENV_PATH:-/storage/brno2/home/jozefsprlak/conda/batl-gpu-cu128}"
module load mambaforge
mamba activate "$ENV_PATH"
PYTHON_EXEC="$ENV_PATH/bin/python"
echo "env: $ENV_PATH"
echo "python: $($PYTHON_EXEC --version)"

echo
echo "===== Torch ====="
$PYTHON_EXEC -u - <<'PY'
import torch
print(f"torch.__version__:        {torch.__version__}")
print(f"torch.version.cuda:       {torch.version.cuda}")
print(f"torch.cuda.is_available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"torch.cuda.device_count:  {torch.cuda.device_count()}")
    print(f"device_name[0]:           {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    print(f"device_capability[0]:     sm_{cap[0]}{cap[1]}")
    free, total = torch.cuda.mem_get_info()
    print(f"gpu_mem_total_gb:         {total / 1024**3:.1f}")
print(f"get_arch_list:            {torch.cuda.get_arch_list()}")
PY

echo
echo "===== FAISS ====="
$PYTHON_EXEC -u - <<'PY'
import faiss
print(f"faiss.__version__:        {faiss.__version__}")
print(f"faiss_gpu_available:      {hasattr(faiss, 'StandardGpuResources')}")
if hasattr(faiss, "StandardGpuResources"):
    res = faiss.StandardGpuResources()
    print("StandardGpuResources():   OK")
PY

echo
echo "===== Smoke: tiny matmul on GPU ====="
$PYTHON_EXEC -u - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("no CUDA")
a = torch.randn(1024, 1024, device="cuda")
b = torch.randn(1024, 1024, device="cuda")
c = (a @ b).sum().item()
print(f"matmul ok, sum={c:.2f}")
PY

echo
echo "===== Done ====="
