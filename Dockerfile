FROM python:3.11.12-slim-bullseye
WORKDIR /app
COPY . /app .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8080
ENTRYPOINT ["streamlit", "run", "webapp.py", "--server.address=0.0.0.0", "--server.port=8080"]
