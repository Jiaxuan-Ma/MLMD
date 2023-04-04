# app/Dockerfile

FROM python:3.9-slim

WORKDIR /Easy to Material Design

COPY . .

# RUN apt-get update -y && apt-get install -y libgomp1

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ \
		&& pip install -r requirements.txt

CMD ["streamlit", "run", "HOME.py", "--server.port", "80"]
