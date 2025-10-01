FROM ubuntu:22.04
WORKDIR /app


RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip install transformers
RUN pip install torch
RUN pip install Pillow
RUN pip install requests
RUN apt-get install -y libopenslide0
RUN pip install openslide-python

COPY . .

CMD ["python3", "main.py", "TCGA-3C-AAAU-01A-01-TS1.2F52DD63-7476-4E85-B7C6-E06092DB6CC1.svs"]

