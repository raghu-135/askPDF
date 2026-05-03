FROM lscr.io/linuxserver/brave:latest

USER root

# Install Python and system dependencies for Playwright
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    xclip \
    # Playwright browser dependencies
    wget \
    gnupg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install FastAPI and dependencies for enhanced CDP
RUN pip install fastapi uvicorn python-multipart requests websockets

# Copy service code
COPY service/ /app/
WORKDIR /app

# Create s6-rc service for capture API (s6-overlay v3 format)
# Service definition
RUN mkdir -p /etc/s6-overlay/s6-rc.d/capture-api

# Create type file (longrun = daemon)
RUN echo 'longrun' > /etc/s6-overlay/s6-rc.d/capture-api/type

# Create dependencies - wait for browser to be ready
RUN printf '%s\n' 'svc-docker' 'svc-xorg' 'svc-selkies' > /etc/s6-overlay/s6-rc.d/capture-api/dependencies

# Create the run script
RUN echo '#!/command/with-contenv bash\nexec 2>&1\necho "[capture-api] Starting..."\n\n# Wait for browser to be ready\nsleep 15\necho "[capture-api] Waiting for browser on port 9222..."\nwhile ! curl -s http://localhost:9222/json/version > /dev/null 2>&1; do\n    echo "[capture-api] Browser not ready yet, waiting..."\n    sleep 3\ndone\necho "[capture-api] Browser is ready!"\n\n# Start the API server\ncd /app\nexport PYTHONPATH=/app\nexec /opt/venv/bin/uvicorn main_playwright:app --host 0.0.0.0 --port 8080 --log-level info --app-dir /app' > /etc/s6-overlay/s6-rc.d/capture-api/run

RUN chmod +x /etc/s6-overlay/s6-rc.d/capture-api/run

# Add service to user bundle (so it starts automatically)
RUN mkdir -p /etc/s6-overlay/s6-rc.d/user/contents.d
RUN touch /etc/s6-overlay/s6-rc.d/user/contents.d/capture-api

# Ensure the captures directory exists with proper permissions
RUN mkdir -p /captures && chown -R abc:abc /captures

# Expose only the Selkies web UI for iframe access.
# CDP and capture API remain internal to the container.
EXPOSE 3000

# The base image already has ENTRYPOINT ["/init"] which runs s6-rc
