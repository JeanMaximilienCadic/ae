FROM python:3.14.0rc1

COPY requirements /
RUN pip install -r requirements.txt

CMD ["python", "-m", "ae.trainers"]

