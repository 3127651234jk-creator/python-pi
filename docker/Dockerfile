FROM python:3.12-slim AS builder
WORKDIR /work
# Build against an ephemeral repository mount. Only outputs staged under
# /opt/evaluator persist; the wheel and caches used in /repo are discarded.
RUN --mount=type=bind,source=.,target=/repo,rw \
    --mount=type=cache,target=/root/.cache/pip \
    cd /repo \
    && python -m venv /opt/evaluator/venv \
    && /opt/evaluator/venv/bin/pip install --no-cache-dir /repo/pi-python \
    && SITE=$(/opt/evaluator/venv/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])") \
    && cp /repo/pi-python/README.md "$SITE/README.md" \
    && mkdir -p /opt/evaluator/root/pi-python/docker \
    && cp /repo/pi-python/pyproject.toml /opt/evaluator/root/pi-python/pyproject.toml \
    && cp -r /repo/pi-python/evals /opt/evaluator/root/pi-python/evals \
    && cp /repo/pi-python/docker/entrypoint.py /opt/evaluator/root/pi-python/docker/entrypoint.py

FROM python:3.12-slim AS evaluator
COPY --from=builder /opt/evaluator/root /repo
COPY --from=builder /opt/evaluator/venv /repo/.venv
ENV PATH="/repo/.venv/bin:${PATH}" PYTHONDONTWRITEBYTECODE=1
WORKDIR /repo/pi-python
# The runner starts as root so it can load evaluator code. Before prompting the
# model, the harness permanently drops to UID 65532. Keep evaluator sources
# root-only so model tools and generated code cannot inspect judges or fixtures.
RUN SITE=$(/repo/.venv/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])") \
    && chmod -R go-rwx "$SITE/pi/evals" \
    && chmod go-rwx /repo/pi-python/evals /repo/pi-python/docker /repo/pi-python/pyproject.toml \
    && mkdir -p /run/pi-eval-secrets \
    && chmod 0700 /run/pi-eval-secrets
ENTRYPOINT ["/repo/.venv/bin/python", "docker/entrypoint.py"]

FROM evaluator AS without_docs
RUN SITE=$(/repo/.venv/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])") \
    && rm -rf "$SITE/README.md" "$SITE/CHANGELOG.md" "$SITE/docs" "$SITE/examples"
ENV PI_EVAL_VARIANT=without_docs

FROM evaluator AS with_docs
ENV PI_EVAL_VARIANT=with_docs
