FROM python:3.11-slim

WORKDIR /app

COPY Code/ /app/

RUN pip install --no-cache-dir dash pandas numpy scikit-learn

EXPOSE 8050

CMD ["python", "app.py"]