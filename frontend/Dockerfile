FROM node:22-bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV HF_HOME=/models/hf \
    TRANSFORMERS_CACHE=/models/hf \
    XDG_CACHE_HOME=/models/cache \
    KOKORO_MODEL_ID=onnx-community/Kokoro-82M-v1.0-ONNX \
    KOKORO_DTYPE=q8 \
    KOKORO_DEVICE=cpu

WORKDIR /app

COPY package.json package-lock.json* ./

RUN npm install --include=optional

COPY src ./src
COPY scripts ./scripts
COPY tsconfig.json ./
COPY next.config.js ./

RUN mkdir -p /models/hf /models/cache \
    && node scripts/prefetch-kokoro.mjs

EXPOSE 3000

CMD ["npm", "run", "dev"]
