# Optional custom training image. The default submit path uses Google's prebuilt
# xgboost-cpu / pytorch-xla images + a python-package sdist (no Docker build needed).
# Build only if you need extra system deps:
#   docker build -t us-docker.pkg.dev/<project>/tactic/train:latest -f vertex/Dockerfile .
FROM us-docker.pkg.dev/vertex-ai/training/pytorch-xla.2-4.py310:latest

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

ENV PYTHONHASHSEED=0
ENTRYPOINT ["python", "-m", "tactic.models.hpo_vertex"]
