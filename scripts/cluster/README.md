# Cluster helper scripts

These scripts contain convenience commands for running Vesuvius jobs on the OSCER cluster.

## Files
- `run_cluster_sweep.sh` — run a validation sweep remotely via SSH.
- `inspect_volume.sh` — inspect predictions for a specific volume and save a slice visualization.
- `launch_training.sh` — launch a training job via SSH.
- `cluster_commands.txt` — scratchpad of common cluster commands.
- `create_repo.sh` — bootstrap a fresh repo layout (meant for local use).

## Usage notes
- Paths assume the repo lives at:
  `/scratch/jowny/vesuvius/vesuviusChallenge/vesuvius-competition`
- Adjust user/paths if you run elsewhere.
- These scripts are intentionally simple; edit as needed for your queue/partition policies.
