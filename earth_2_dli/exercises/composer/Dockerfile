FROM nvcr.io/nvidia/modulus/modulus:24.12

# Typical DLI jupyter notebook setup boilerplate
COPY composer/jupyter/ /root/.jupyter/
COPY composer/jupyter/bashrc /root/.bashrc
ENV SHELL /bin/bash
EXPOSE 8888
WORKDIR /workspace/exercises

# Environment variables
ENV HDF5_USE_FILE_LOCKING=FALSE
ENV LOGURU_LEVEL=INFO
ENV EARTH2STUDIO_CACHE=/workspace/data/earth2cache
ENV EARTH2STUDIO_PACKAGE_TIMEOUT=600
ENV NGC_CLI_ORG=no-org
ENV NGC_CLI_TEAM=no-team

# Dependencies
RUN pip install --upgrade ipywidgets jupyterlab==4.2.0 jupyterlab-widgets jupyter-archive jupyter-resource-usage
COPY composer/requirements.txt ..
# RUN pip install --upgrade pip && pip install -r ../requirements.txt
RUN pip install --upgrade "pip<25" && pip install --prefer-binary -r ../requirements.txt


# Download data
COPY composer/fetch_data.py ..
RUN python3 ../fetch_data.py

# RUN mkdir -p /workspace/data/earth2cache
# WORKDIR /workspace/data/earth2cache
# RUN wget -q https://dli-lms.s3.us-east-1.amazonaws.com/data/x-fx-82-v1/cache4gtc.tar.gz
# RUN tar xzf cache4gtc.tar.gz --strip-components=1 && rm -f cache4gtc.tar.gz
# RUN ls -lah

# Create working directory parent to add repo and execute entryfile
COPY composer/entrypoint.sh ..
ENTRYPOINT ["/workspace/entrypoint.sh"]
