FROM continuumio/miniconda3

WORKDIR /app/GenoPheno
COPY installation/environment.yaml .

RUN mkdir src && apt -y update && apt -y --no-install-recommends install gcc python3-dev xvfb libxtst6 libxcomposite1 libxcursor1 libxss1 libpci3 libasound2 libgtk2.0-0 libgconf-2-4 libnss3 && conda env create -f environment.yaml && conda clean -afy && find /opt/conda/ -follow -type f -name '*.a' -delete \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
    && find /opt/conda/ -follow -type f -name '*.js.map' -delete && rm -rf /var/lib/apt/lists/*

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "genopheno", "/bin/bash", "-c"]

RUN pip install joblib && pip install xlwt && pip install psutil &&  apt -y remove gcc && cd /opt/conda/envs/genopheno/lib && npm install electron --save-dev
ENV QT_QPA_PLATFORM='offscreen'

COPY src/ src

ENTRYPOINT ["conda", "run", "-n", "genopheno", "python", "src/run.py"]

