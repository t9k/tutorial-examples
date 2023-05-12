FROM deepspeed/deepspeed:v072_torch112_cu117

RUN apt-get update && apt-get install -yq --no-install-recommends \
  openssh-server

RUN pip install --no-cache-dir -i https://pypi.douban.com/simple/ -U \
    deepspeed

RUN mkdir /run/sshd
RUN chown root:root /usr/lib

ENTRYPOINT /usr/sbin/sshd -D && bash
