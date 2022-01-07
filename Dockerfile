FROM python:3.8.0-slim

RUN apt-get update
RUN apt-get install git ffmpeg libsm6 libxext6 openssh-client -y
RUN apt install build-essential -y --no-install-recommends

# Install ssh client and git
#RUN apt-get install openssh-client -y

# Download public key for github.com
RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

# Clone and setup
WORKDIR /
RUN --mount=type=ssh git clone git@github.com:renjithsasidharan/meeter-reader-inference.git
WORKDIR ./meeter-reader-inference
RUN pip install -r requirements.txt
WORKDIR ./lanms
RUN mv Makefile.win Makefile
RUN make
WORKDIR /meeter-reader-inference

EXPOSE 8501
ENTRYPOINT ["streamlit","run"]
CMD ["app.py"]