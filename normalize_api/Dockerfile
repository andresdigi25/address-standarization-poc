
FROM python:3.10

WORKDIR /app

COPY . /app

RUN pip install fastapi uvicorn pydantic

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
