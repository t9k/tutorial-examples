FROM t9kpublic/pytorch-1.13.0:sdk-0.5.2

USER root
RUN apt-get update && apt-get install -yq --no-install-recommends \
  openssh-client \
  openssh-server \
  && apt-get clean && \
  rm -rf /var/lib/apt/lists/*
COPY ssh_server.sh /t9k/app/ssh_server.sh

USER t9kuser
