FROM continuumio/miniconda3

WORKDIR /app/GenoPheno
COPY installation/environment.yaml .

# https://github.com/plotly/orca/issues/150
RUN mkdir -p src data/configs/scikit_models && apt -y update && apt -y --no-install-recommends install procps gcc poppler-utils python3-dev xvfb wget xvfb xauth libgtk2.0-0 libxtst6 libxss1 libgconf-2-4 libnss3 libasound2 libxtst6 libxcomposite1 libxcursor1 libxss1 libpci3 libasound2 libgtk2.0-0 libgconf-2-4 libnss3 && conda env create -f environment.yaml && conda clean -afy && find /opt/conda/ -follow -type f -name '*.a' -delete \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
    && find /opt/conda/ -follow -type f -name '*.js.map' -delete && rm -rf /var/lib/apt/lists/* \
    && printf '#!/bin/bash \nxvfb-run --auto-servernum --server-args "-screen 0 640x480x24" /opt/conda/envs/genopheno/lib/orca_app/orca "$@"' > /opt/conda/envs/genopheno/bin/orca  \
    && chmod +x /opt/conda/envs/genopheno/bin/orca




#     && chmod -R 777 /opt/conda/envs/genopheno \
#     && mkdir -p /opt/orca \
#     && cd /opt/orca \
#     && wget https://github.com/plotly/orca/releases/download/v1.2.1/orca-1.2.1-x86_64.AppImage \
#     && chmod +x orca-1.2.1-x86_64.AppImage \
#     && ./orca-1.2.1-x86_64.AppImage --appimage-extract \
#     && rm orca-1.2.1-x86_64.AppImage \
#     && printf '#!/bin/bash \nxvfb-run --auto-servernum --server-args "-screen 0 640x480x24" /opt/orca/squashfs-root/app/orca "$@"' > /opt/conda/envs/genopheno/bin/orca \
#     && chmod +x /opt/conda/envs/genopheno/bin/orca \
#     && chmod -R 777 /root \
#     && chmod -R 777 /opt/orca










# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "genopheno", "/bin/bash", "-c"]

RUN pip install pdf2image && pip install joblib && pip install xlwt && pip install psutil &&  apt -y remove gcc && cd /opt/conda/envs/genopheno/lib && npm install electron --save-dev 
ENV QT_QPA_PLATFORM='offscreen'

COPY src/ src
COPY data/configs/scikit_models/ data/configs/scikit_models

# ENTRYPOINT ["conda", "run", "-n", "genopheno", "python", "src/run.py"]

