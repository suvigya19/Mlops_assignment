FROM python:3.9-slim

WORKDIR /Users/suvigyasharma/Mlops_assignment/Assignment5_6
COPY . .

RUN apt-get update && apt-get install -y build-essential libssl-dev libffi-dev python3-dev
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 80
CMD ["python3", "train.py"]
