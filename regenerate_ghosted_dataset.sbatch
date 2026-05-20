#!/bin/bash
#
# Regenerate the full ghosted MP-20 dataset (val, test, train) after the
# periodicity fix in voronoi_weighted_noise.py.
#
# Sequentially runs: val -> test -> train.
# CPU-only job (Voronoi tessellation via freud, no GPU needed).
#
# Submit with:  sbatch regenerate_ghosted_dataset.sbatch
#

#SBATCH --account=torch_pr_148_general
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=ghost-regen-j05
#SBATCH --output=ghost-regen-j05.out

set -euo pipefail

# ---- Paths (adjust if your HPC layout differs) -------------------------------
WORKDIR=/scratch/kd2862/omatg-ghosted/ghosting-repo
OVERLAY=/scratch/kd2862/omatg-ghosted/overlay-15GB-500K.ext3
IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif

# Name of the new processed_datasets/ subdirectory to write to.
# Periodicity fix + 0.05 supercell jitter.
RUN_FOLDER=mp_20_periodic_fix_j05

# ---- Logging -----------------------------------------------------------------
echo "==== ghost-regen-j05 ===="
echo "Host:      $(hostname)"
echo "Date:      $(date)"
echo "Workdir:   ${WORKDIR}"
echo "Overlay:   ${OVERLAY}"
echo "Image:     ${IMAGE}"
echo "Run name:  ${RUN_FOLDER}"
echo "Output to: ${WORKDIR}/processed_datasets/${RUN_FOLDER}/"
echo "========================="

cd "${WORKDIR}"

# Use scratch for cache/tmp to avoid home-quota issues (mirrors run_benchmarks.sbatch)
export TMPDIR=/scratch/kd2862/tmp
mkdir -p "${TMPDIR}"

# ---- Run val -> test -> train (sequential, single Singularity exec each) -----
# Using --nv is unnecessary here (CPU-only); skip it for cleaner output.
for SPLIT in val test train; do
    echo ""
    echo "---- Generating split: ${SPLIT} ($(date)) ----"
    SECONDS=0
    singularity exec \
        --overlay "${OVERLAY}":rw \
        "${IMAGE}" \
        /bin/bash -c "source /ext3/env.sh && cd ${WORKDIR} && python pete_data_gen.py ${SPLIT} ${RUN_FOLDER}"
    echo "---- Done split: ${SPLIT} (elapsed ${SECONDS}s) ----"
done

echo ""
echo "==== All splits complete: $(date) ===="
ls -lh "${WORKDIR}/processed_datasets/${RUN_FOLDER}/"
