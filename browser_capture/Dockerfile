FROM lscr.io/linuxserver/brave:1.91.175-ls106@sha256:4757ede17ac763746b4863a63124a3147084461591bc53679ddb67bf1fdfd755

USER root

# The July 2026 linuxserver/brave base defaults to PixelFlux/Wayland, where
# Brave 1.92 aborts before exposing CDP in this container. Keep capture on the
# Xorg path unless we intentionally revisit the base-image Wayland behavior.
ENV PIXELFLUX_WAYLAND=false

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    xclip \
    libpango-1.0-0 \
    libharfbuzz0b \
    libpangoft2-1.0-0 \
    libfontconfig1 \
    libcairo2 \
    libgdk-pixbuf-xlib-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install FastAPI and dependencies for direct CDP
RUN pip install fastapi uvicorn python-multipart requests websockets weasyprint pypdf

# ChromeDriver will be downloaded automatically by webdriver-manager

# Copy service code
COPY service/ /app/
WORKDIR /app

# Disable the base desktop autostart for Brave. The capture path needs CDP to be
# deterministic and observable, so we supervise Brave explicitly below.
RUN printf '%s\n' '#!/bin/bash' 'exit 0' > /defaults/autostart \
    && printf '%s\n' '#!/bin/bash' 'exit 0' > /defaults/autostart_wayland

# Create s6-rc service for Brave with Chrome DevTools Protocol enabled.
RUN mkdir -p /etc/s6-overlay/s6-rc.d/browser-cdp
RUN echo 'longrun' > /etc/s6-overlay/s6-rc.d/browser-cdp/type
RUN printf '%s\n' 'svc-de' 'svc-selkies' > /etc/s6-overlay/s6-rc.d/browser-cdp/dependencies
RUN cat <<'EOF' > /etc/s6-overlay/s6-rc.d/browser-cdp/run
#!/command/with-contenv bash
exec 2>&1

echo "[browser-cdp] Starting Brave with CDP..."

export DISPLAY="${DISPLAY:-:1}"
BRAVE_DISPLAY_ARGS=()

if [ "${PIXELFLUX_WAYLAND}" = "true" ]; then
    export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/config/.XDG}"
    export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-1}"
    SOCKET_PATH="${XDG_RUNTIME_DIR}/${WAYLAND_DISPLAY}"
    echo "[browser-cdp] Waiting for Wayland socket at ${SOCKET_PATH}..."
    while [ ! -e "${SOCKET_PATH}" ]; do
        sleep 0.5
    done

    echo "[browser-cdp] Wayland socket is ready"
    BRAVE_DISPLAY_ARGS=(--enable-features=UseOzonePlatform --ozone-platform=wayland)
else
    echo "[browser-cdp] Waiting for X display ${DISPLAY}..."
    until s6-setuidgid abc xset q >/dev/null 2>&1; do
        sleep 0.5
    done

    echo "[browser-cdp] X display is ready"
fi
sleep 2
mkdir -p /config/.config/BraveSoftware/Brave-Browser
mkdir -p /config/.cache/BraveSoftware/Brave-Browser
chown -R abc:abc /config/.config/BraveSoftware
chown -R abc:abc /config/.cache
rm -f /config/.config/BraveSoftware/Brave-Browser/Singleton*

exec s6-setuidgid abc /usr/bin/brave-browser \
    --no-first-run \
    --no-sandbox \
    --password-store=basic \
    --simulate-outdated-no-au='Tue, 31 Dec 2099 23:59:59 GMT' \
    --disable-crash-reporter \
    --start-maximized \
    --test-type \
    --user-data-dir=/config/.config/BraveSoftware/Brave-Browser \
    "${BRAVE_DISPLAY_ARGS[@]}" \
    --remote-debugging-address=127.0.0.1 \
    --remote-debugging-port=9222 \
    --no-default-browser-check \
    about:blank
EOF
RUN chmod +x /etc/s6-overlay/s6-rc.d/browser-cdp/run

# Create s6-rc service for capture API (s6-overlay v3 format)
# Service definition
RUN mkdir -p /etc/s6-overlay/s6-rc.d/capture-api

# Create type file (longrun = daemon)
RUN echo 'longrun' > /etc/s6-overlay/s6-rc.d/capture-api/type

# Create dependencies - wait for browser-cdp to be supervised before probing CDP.
RUN printf '%s\n' 'svc-docker' 'svc-xorg' 'svc-selkies' 'browser-cdp' > /etc/s6-overlay/s6-rc.d/capture-api/dependencies

# Create the run script
RUN cat <<'EOF' > /etc/s6-overlay/s6-rc.d/capture-api/run
#!/command/with-contenv bash
exec 2>&1

echo "[capture-api] Starting..."
echo "[capture-api] Waiting for browser on port 9222..."
while ! curl -s http://localhost:9222/json/version > /dev/null 2>&1; do
    if pgrep -a brave >/tmp/browser-cdp-processes 2>/dev/null; then
        echo "[capture-api] Browser not ready yet; Brave process: $(head -n 1 /tmp/browser-cdp-processes)"
    else
        echo "[capture-api] Browser not ready yet; no Brave process found"
    fi

    if command -v ss >/dev/null 2>&1; then
        ss -ltnp 2>/dev/null | grep ':9222' || true
    fi
    sleep 3
done
echo "[capture-api] Browser is ready!"

cd /app
export PYTHONPATH=/app
exec /opt/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8080 --log-level info --app-dir /app
EOF

RUN chmod +x /etc/s6-overlay/s6-rc.d/capture-api/run

# Add service to user bundle (so it starts automatically)
RUN mkdir -p /etc/s6-overlay/s6-rc.d/user/contents.d
RUN touch /etc/s6-overlay/s6-rc.d/user/contents.d/browser-cdp
RUN touch /etc/s6-overlay/s6-rc.d/user/contents.d/capture-api

# Ensure the captures directory exists with proper permissions
RUN mkdir -p /captures && chown -R abc:abc /captures

# Expose only the Selkies web UI for iframe access.
# CDP and capture API remain internal to the container.
EXPOSE 3000

# The base image already has ENTRYPOINT ["/init"] which runs s6-rc
