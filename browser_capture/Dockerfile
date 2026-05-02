FROM lscr.io/linuxserver/brave:latest

USER root

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Playwright and FastAPI
RUN pip install playwright fastapi uvicorn python-multipart

# Install Chromium browser for Playwright
RUN playwright install chromium

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
RUN echo '#!/command/with-contenv bash\n\
exec 2>&1\n\
echo "[capture-api] Starting..."\n\
\n\
# Wait for browser CDP to be ready\n\
sleep 15\n\
echo "[capture-api] Waiting for CDP on port 9222..."\n\
while ! curl -s http://localhost:9222/json/version > /dev/null 2>&1; do\n\
    echo "[capture-api] CDP not ready yet, waiting..."\n\
    sleep 3\n\
done\n\
echo "[capture-api] CDP is ready!"\n\
\n\
# Start the API server\n\
cd /app\n\
export PYTHONPATH=/app\n\
exec /opt/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8080 --log-level info --app-dir /app' > /etc/s6-overlay/s6-rc.d/capture-api/run

RUN chmod +x /etc/s6-overlay/s6-rc.d/capture-api/run

# Add service to user bundle (so it starts automatically)
RUN mkdir -p /etc/s6-overlay/s6-rc.d/user/contents.d
RUN touch /etc/s6-overlay/s6-rc.d/user/contents.d/capture-api

# Ensure the captures directory exists with proper permissions
RUN mkdir -p /captures && chown -R abc:abc /captures

# Expose ports: 3000/3001 (browser), 9222 (CDP), 8080 (API)
EXPOSE 3000 3001 9222 8080

# The base image already has ENTRYPOINT ["/init"] which runs s6-rc
