FROM mcr.microsoft.com/devcontainers/python:2-3.14-trixie

COPY requirements.txt requirements_demo.txt /tmp/pip-tmp/
RUN if [ -f "/tmp/pip-tmp/requirements.txt" ]; then pip install --system -r /tmp/pip-tmp/requirements.txt; fi \
    && if [ -f "/tmp/pip-tmp/requirements_demo.txt" ]; then pip install --system -r /tmp/pip-tmp/requirements_demo.txt; fi \
    && rm -rf /tmp/pip-tmp

# [Optional] Uncomment this section to install additional OS packages.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends libgl1
