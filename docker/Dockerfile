FROM dustynv/jetson-voice:r32.6.1

RUN git clone https://github.com/Armandpl/jetson_denoiser && \
    cd jetson_denoiser && pip install .

RUN python -m pip install --upgrade pip
RUN pip install soxr
