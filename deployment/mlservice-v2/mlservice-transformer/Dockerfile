FROM python:3.8-slim

COPY t9k-sdk.tar.gz t9k-sdk.tar.gz
COPY server.py server.py

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    numpy \
    pillow \
    requests
RUN pip install t9k-sdk.tar.gz  -i https://pypi.tuna.tsinghua.edu.cn/simple

ENTRYPOINT ["python", "server.py"]
