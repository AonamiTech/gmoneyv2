FROM node:22-alpine AS dependencies
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

FROM node:22-alpine AS builder
WORKDIR /app
COPY --from=dependencies /app/node_modules ./node_modules
COPY frontend ./
RUN npm run typecheck && npm run build

FROM node:22-alpine AS runner
ARG GMONEY_BUILD_REVISION=unknown
ARG GMONEY_REQUIRE_BUILD_REVISION=0
LABEL org.opencontainers.image.revision=${GMONEY_BUILD_REVISION}
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    HOSTNAME=0.0.0.0 \
    PORT=3000
WORKDIR /app
RUN addgroup --system --gid 10001 web && adduser --system --uid 10001 --ingroup web web
COPY infra/docker/write-release-manifest.sh /tmp/write-release-manifest.sh
RUN sh /tmp/write-release-manifest.sh && rm /tmp/write-release-manifest.sh
COPY --from=builder --chown=web:web /app/.next/standalone ./
COPY --from=builder --chown=web:web /app/.next/static ./.next/static
COPY --chown=web:web frontend/docker-entrypoint.mjs ./docker-entrypoint.mjs
USER web
EXPOSE 3000
CMD ["sh", "-c", "node /app/docker-entrypoint.mjs && exec node server.js"]
