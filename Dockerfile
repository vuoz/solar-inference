FROM --platform=linux/amd64 rust:latest AS base

WORKDIR /build

RUN apt-get update && apt-get install -y \
    libtorch-dev \
    libstdc++-12-dev\
    curl \
    unzip \
    clang \
    cmake \
    build-essential \
    pkg-config \
    libssl-dev \
    wget\
    cmake 

RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcpu.zip -O libtorch.zip
RUN unzip -o libtorch.zip



# Copy project files
COPY . /build

FROM base AS build
COPY --from=base . /build

ENV LIBTORCH_INCLUDE="/build/libtorch/"
ENV LIBTORCH_LIB="/build/libtorch/"
ENV LIBTORCH="/build/libtorch/"
ENV LD_LIBRARY_PATH="/build/libtorch/lib:$LD_LIBRARY_PATH"
ENV LIBTORCH_CXX11_ABI=1




RUN cargo build --release 

FROM debian:bullseye-slim AS runtime

RUN apt-get update && apt-get install -y \
    libssl-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=base /build/target/release/solar-inference /app/solar-inference
COPY --from=base /build/libtorch /usr/build/libtorch




EXPOSE 4444
CMD ["./solar-inference"]

