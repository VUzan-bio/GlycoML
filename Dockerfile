FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build

FROM python:3.10-slim
WORKDIR /app

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY backend/ /app/backend/
COPY data/ /app/data/
COPY outputs/ /app/outputs/
COPY --from=frontend-build /app/frontend/dist /app/frontend/dist

EXPOSE 8000
ENV USE_MODEL_PREDICTIONS=true
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
