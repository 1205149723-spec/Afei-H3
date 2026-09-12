FROM nvidia/cuda:13.0.2-devel-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv python3-pip ffmpeg gcc g++ ninja-build git curl ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace/HailuoH3-8775-AutoDL
COPY . .
RUN bash install_autodl.sh
EXPOSE 6006
CMD ["bash", "start_autodl.sh"]
