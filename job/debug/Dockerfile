FROM t9kpublic/pytorch-1.13.0:sdk-0.5.2

# 原镜像的默认用户和用户组为 t9kuser:t9kuser (1000:1000)
USER root

RUN apt-get update && apt-get install -yq --no-install-recommends \
  openssh-client \
  openssh-server \
  && apt-get clean && \
  rm -rf /var/lib/apt/lists/*
COPY ssh_server.sh /t9k/app/ssh_server.sh
# propagate all env vars
SHELL ["/bin/bash", "-c"]
RUN echo "export \$(cat /proc/1/environ | tr '\0' '\n' | xargs)" >> /etc/profile

RUN chown -R 1000:1000 /t9k/app && chmod 755 /t9k/app
USER t9kuser
