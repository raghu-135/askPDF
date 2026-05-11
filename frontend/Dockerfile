FROM node:22-bookworm-slim AS base

ENV NEXT_TELEMETRY_DISABLED=1 \
    HF_HOME=/models/hf \
    TRANSFORMERS_CACHE=/models/hf \
    XDG_CACHE_HOME=/models/cache \
    KOKORO_MODEL_ID=onnx-community/Kokoro-82M-v1.0-ONNX \
    KOKORO_DTYPE=q8 \
    KOKORO_DEVICE=cpu

WORKDIR /app

FROM base AS deps

COPY package.json package-lock.json ./

RUN npm ci

FROM base AS dev

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=deps /app/node_modules ./node_modules
COPY package.json package-lock.json ./

EXPOSE 3000

CMD ["npm", "run", "dev"]

FROM deps AS builder

ARG NEXT_PUBLIC_API_URL
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}

COPY next.config.js ./
COPY next-env.d.ts ./
COPY tsconfig.json ./
COPY src ./src
COPY scripts ./scripts

RUN mkdir -p /models/hf /models/cache \
    && node scripts/prefetch-kokoro.mjs \
    && npm run build

FROM node:22-bookworm-slim AS runtime

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0 \
    HF_HOME=/models/hf \
    TRANSFORMERS_CACHE=/models/hf \
    XDG_CACHE_HOME=/models/cache \
    KOKORO_MODEL_ID=onnx-community/Kokoro-82M-v1.0-ONNX \
    KOKORO_DTYPE=q8 \
    KOKORO_DEVICE=cpu

WORKDIR /app

COPY --from=builder /models /models
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

EXPOSE 3000

CMD ["node", "server.js"]
