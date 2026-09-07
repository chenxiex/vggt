FROM ghcr.io/chenxiex/devcontainer-templates/devcontainer-python-uv-3-trixie:latest

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends libgl1 libboost-filesystem-dev libboost-system-dev libeigen3-dev libpcl-dev cmake