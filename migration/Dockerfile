FROM askpdf-rag-service:latest

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY rag_service/app app
COPY supabase/migrations /supabase/migrations

ENTRYPOINT ["python", "-m", "app.tools.db_migration_cli"]
