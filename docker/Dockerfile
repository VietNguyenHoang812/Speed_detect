FROM nvcr.io/nvidia/l4t-pytorch:r32.5.0-pth1.7-py3
COPY requirements.txt /requirements.txt
RUN python3 -m pip install -r /requirements.txt
RUN export PATH=/usr/local/cuda-10.2/bin${PATH:+:${PATH}}
RUN LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH
RUN python3 -m pip install pycuda --user
RUN apt install python3-seaborn
RUN python3 -m pip install -U jetson-stats==3.1.4

