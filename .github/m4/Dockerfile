# One image per lane job; every episode runs in its own container from it.
# The container gives each agent its own localhost (the node's HTTP port 4243,
# the edge's 4242, the API's 8080 and the mock logshipper's 18080 are fixed in
# the engine, ciris-server and the harness) and its own writable copy of /src
# (data/, logs/, qa_reports/, CIRIS_HOME), so concurrent episodes share nothing
# but the kernel and the network uplink.
#
# Build context (prepared by the workflow): agent/ = the harness checkout,
# extra/ = optional constraints.txt and candidate ciris_server-*.whl.
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1
RUN apt-get update -o Acquire::Retries=3 \
 && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3.12-dev build-essential git curl ca-certificates procps jq \
      libtss2-tctildr0 libtss2-esys-3.0.2-0 libtss2-mu-4.0.1-0 \
 && rm -rf /var/lib/apt/lists/*
RUN python3.12 -m venv /venv
ENV PATH=/venv/bin:$PATH
COPY agent /src
COPY extra /extra
WORKDIR /src
RUN set -eu; \
    python -m pip install --upgrade pip; \
    C=""; if [ -f /extra/constraints.txt ]; then C="-c /extra/constraints.txt"; fi; \
    pip install $C -e .; \
    pip install $C -r requirements-dev.txt; \
    for w in /extra/ciris_server-*.whl; do \
      if [ -e "$w" ]; then pip install --force-reinstall --no-deps "$w"; echo "candidate ciris-server installed: $(basename "$w")"; fi; \
    done
