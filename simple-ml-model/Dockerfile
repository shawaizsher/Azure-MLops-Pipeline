FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=80 \
    STREAMLIT_SERVER_HEADLESS=true

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY . .

# App Service for Containers routes to port 80 by default.
EXPOSE 80

CMD ["sh", "-c", "python -m streamlit run app.py --server.address 0.0.0.0 --server.port ${PORT} --server.headless true"]
