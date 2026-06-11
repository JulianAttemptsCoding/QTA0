# Vertex AI — heavy compute

All GPU/large-CPU work (neural-expert HPO, 5-seed training, CPCV path sweeps) runs on
Vertex AI custom-jobs. Light work (ingestion, diagnostics, accounting) runs locally.

## Config (from configs/v1.yaml[vertex], env-overridable)

| key | value |
|-----|-------|
| project_id | `project-c779f701-1a49-4a58-b54` |
| region | `us-central1` |
| account | `juliansjuan08@gmail.com` |
| bucket | `gs://gmda-vertex-c779f701-uscentral1` |
| fallback bucket | `gs://gmda-vertex-c779f701-uscentral1-juliansjuan08` |
| machine (full) | `n1-standard-16` |
| machine (smoke) | `n1-standard-4` |
| image cpu/smoke | `us-docker.pkg.dev/vertex-ai/training/xgboost-cpu.2-1:latest` |
| image gpu | `us-docker.pkg.dev/vertex-ai/training/pytorch-xla.2-4.py310:latest` |

## One-time auth

```
gcloud auth login juliansjuan08@gmail.com
gcloud config set project project-c779f701-1a49-4a58-b54
gcloud config set ai/region us-central1
```

## Submit a job

```
# smoke (n1-standard-4, cpu image)
python vertex/submit.py --module tactic.models.hpo_vertex --smoke

# full neural HPO on GPU image
python vertex/submit.py --module tactic.models.hpo_vertex --gpu --args expert=f4 refit=2021
```

`submit.py` builds the sdist, uploads it to `gs://.../packages/`, and launches the custom
job with the exact `gcloud ai custom-jobs create` pattern for this project. `vizier_spec.yaml`
holds the HPO search space. `Dockerfile` is only needed for extra system deps.
