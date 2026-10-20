# syntax=docker/dockerfile:1
# Linux ROCm userspace only. The host keeps its own kernel driver and firmware.
FROM rocm/dev-ubuntu-24.04:7.2.3-complete@sha256:ec1b59bf75ec1122e7a091c0be82301ba12458b038499ef96a5c115876bd78d2
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 LANG=C.UTF-8
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl git libatomic1 libgomp1 python3-venv \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/strata
COPY requirements.txt ./
COPY . .
RUN python3 -m venv .venv \
    && if test -d vendor/wheelhouse; then \
         (cd vendor && sha256sum -c SHA256SUMS) \
         && .venv/bin/pip install --no-cache-dir --no-index --find-links=vendor/wheelhouse -r requirements.txt -r requirements-build.txt; \
       else .venv/bin/pip install --no-cache-dir -r requirements.txt -r requirements-build.txt; fi
ENV PATH=/opt/strata/.venv/bin:/opt/rocm/bin:${PATH}
ARG HIP_ARCH=gfx1200
ARG BUILD_JOBS=4
ARG SOURCE_REVISION=unknown
LABEL org.opencontainers.image.title="Strata AMD NAS" \
      org.opencontainers.image.revision=${SOURCE_REVISION}
# Use the exact dependency named by third_party/ggml/VERSION.txt, without cloning history.
RUN mkdir -p /opt/llama.cpp \
    && if test -f vendor/llama.cpp-pinned.tar.gz; then \
         echo 'c076d7534afa0e5d0ec2a0d425b11e791c16f3de0d727221aea071cef156a280  vendor/llama.cpp-pinned.tar.gz' | sha256sum -c - \
         && tar -xzf vendor/llama.cpp-pinned.tar.gz --strip-components=1 -C /opt/llama.cpp; \
       else curl --fail --location --retry 3 \
       https://codeload.github.com/ggml-org/llama.cpp/tar.gz/3cf03257f219afbe7334045ff7c6a06ac68c627d \
       | tar -xz --strip-components=1 -C /opt/llama.cpp; fi \
    && cmake -S /opt/strata -B /opt/strata/build-hip -G Ninja \
       -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
       -DCMAKE_HIP_ARCHITECTURES=${HIP_ARCH} \
       -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
       -DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm -DCMAKE_PREFIX_PATH=/opt/rocm \
       -DSTRATA_PREFILL_MMQ=ON -DSTRATA_BUILD_TESTS=OFF \
       -DSTRATA_GGML_DIR=/opt/llama.cpp \
    && cmake --build /opt/strata/build-hip --target strata strata-device -j${BUILD_JOBS} \
    && printf '%s\n' "${SOURCE_REVISION}" > /opt/strata/SOURCE_REVISION \
    && printf '%s\n' "${SOURCE_REVISION}" > /opt/strata/COMMIT
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=900s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=4).read()"
ENTRYPOINT ["/opt/strata/.venv/bin/python", "/opt/strata/deploy/nas/entrypoint.py"]
