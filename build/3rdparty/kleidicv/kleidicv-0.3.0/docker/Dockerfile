# SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

FROM ubuntu:24.04

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get -y --no-install-recommends install \
      ca-certificates \
      ccache \
      cmake \
      doxygen \
      g++-aarch64-linux-gnu \
      gcc-aarch64-linux-gnu \
      git \
      gpg \
      graphviz \
      less \
      libnuma1 \
      liburing2 \
      lsb-release \
      ninja-build \
      patch \
      pipx \
      python3 \
      shellcheck \
      software-properties-common \
      wget

# Install latest version of most LLVM tools except clang-format and clang-tidy
# because we prioritise stability when it comes to how we format our code and
# which cland-tidy checks we expect.
ENV LLVM_VERSION=20
ENV CLANG_FORMAT_VERSION=19
ENV CLANG_TIDY_VERSION=19

RUN wget -q https://apt.llvm.org/llvm-snapshot.gpg.key -O /etc/apt/trusted.gpg.d/apt.llvm.org.asc \
    && echo "ce6eee4130298f79b0e0f09a89f93c1bc711cd68e7e3182d37c8e96c5227e2f0 /etc/apt/trusted.gpg.d/apt.llvm.org.asc" | sha256sum -c - \
    && . /etc/os-release \
    && add-apt-repository "deb http://apt.llvm.org/${VERSION_CODENAME}/ llvm-toolchain-${VERSION_CODENAME} main" \
    && add-apt-repository "deb http://apt.llvm.org/${VERSION_CODENAME}/ llvm-toolchain-${VERSION_CODENAME}-${CLANG_FORMAT_VERSION} main" \
    && add-apt-repository "deb http://apt.llvm.org/${VERSION_CODENAME}/ llvm-toolchain-${VERSION_CODENAME}-${CLANG_TIDY_VERSION} main" \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get -y --no-install-recommends install \
        clang-${LLVM_VERSION} \
        clang-format-${CLANG_FORMAT_VERSION} \
        clang-tidy-${CLANG_TIDY_VERSION} \
        clang-tools-${LLVM_VERSION} \
        libclang-rt-${LLVM_VERSION}-dev \
        lld-${LLVM_VERSION} \
        llvm-${LLVM_VERSION}

# Populate the contents of libclang-rt-dev:arm64 on an x86 host.
# Package installation fails, so instead the package is downloaded and extracted into the root filesystem.
RUN if [ $(dpkg --print-architecture) = amd64 ]; then \
    dpkg --add-architecture arm64 \
    && (apt-get -y update || true) \
    && cd /tmp \
    && apt-get -y download libclang-rt-${LLVM_VERSION}-dev:arm64 \
    && dpkg-deb -x libclang-rt*.deb / \
    && rm -rf libclang-rt* \
    && dpkg --remove-architecture arm64 \
    ; fi

RUN pipx install cpplint==1.6.1
RUN pipx install gcovr==6.0
RUN pipx install reuse==3.0.0

ENV PATH=${PATH}:/usr/lib/llvm-${LLVM_VERSION}/bin:/root/.local/bin
ENV CC=clang-${LLVM_VERSION}
ENV CXX=clang++-${LLVM_VERSION}

ARG OPENCV_VERSION=4.10.0
RUN wget \
  https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.tar.gz \
  -O /opt/opencv-${OPENCV_VERSION}.tar.gz
RUN wget \
  https://github.com/opencv/opencv_extra/archive/refs/tags/${OPENCV_VERSION}.tar.gz \
  -O /opt/opencv-extra-${OPENCV_VERSION}.tar.gz
