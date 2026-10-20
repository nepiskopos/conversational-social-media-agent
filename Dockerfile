FROM python:3.12-slim

WORKDIR /app

# Log lines reach `docker compose logs` as they are written, not when a buffer fills (or never,
# if the process dies first).
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as an unprivileged user that can't change the app's code or settings
# (.chainlit/config.toml). Chainlit writes only .files and .chainlit/translations. Bytecode
# isn't cached, as the user can't write it.
ENV PYTHONDONTWRITEBYTECODE=1
RUN useradd --create-home --uid 10001 mackie && mkdir -p .files \
    && chown -R mackie .files .chainlit/translations
USER mackie

EXPOSE 8000

CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]
