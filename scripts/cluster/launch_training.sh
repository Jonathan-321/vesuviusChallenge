#!/bin/bash
# Launch 60-epoch training on A100

ssh jowny@schooner.oscer.ou.edu << 'EOF'
set -e

cd /scratch/jowny/vesuvius/vesuviusChallenge/vesuvius-competition
export PYTHONPATH=$PWD:$PWD/src
source venv/bin/activate

# Create SLURM script for A100 training
cat > train_60epoch.sbatch << 'SBATCH'
#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --job-name=vesuvius60
#SBATCH --output=logs/train_60epoch_%j.out
#SBATCH --error=logs/train_60epoch_%j.err

echo "Starting 60-epoch training on A100"
echo "Node: $SLURM_NODELIST"
nvidia-smi

cd /scratch/jowny/vesuvius/vesuviusChallenge/vesuvius-competition
export PYTHONPATH=$PWD:$PWD/src
source venv/bin/activate

python train.py --config configs/experiments/surface_unet3d_tuned.yaml
SBATCH

# Submit the job
mkdir -p logs
echo "Submitting training job..."
sbatch train_60epoch.sbatch

# Also create an interactive training script for quick testing
cat > run_training_interactive.sh << 'SCRIPT'
#!/bin/bash
# Run training interactively (for testing)
cd /scratch/jowny/vesuvius/vesuviusChallenge/vesuvius-competition
export PYTHONPATH=$PWD:$PWD/src
source venv/bin/activate
srun -p gpu --gpus=1 --constraint=a100 --mem=64G --time=24:00:00 --pty \
    python train.py --config configs/experiments/surface_unet3d_tuned.yaml
SCRIPT
chmod +x run_training_interactive.sh

echo "Created train_60epoch.sbatch and run_training_interactive.sh"
echo "To monitor: squeue -u jowny"
EOF