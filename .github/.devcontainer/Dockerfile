FROM mcr.microsoft.com/devcontainers/python:3.12

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends libgl1

COPY requirements.txt requirements_demo.txt /tmp/pip-tmp/
RUN if [ -f "/tmp/pip-tmp/requirements.txt" ]; then pip install -r /tmp/pip-tmp/requirements.txt; fi \
    && if [ -f "/tmp/pip-tmp/requirements_demo.txt" ]; then pip install -r /tmp/pip-tmp/requirements_demo.txt; fi \
    && rm -rf /tmp/pip-tmp