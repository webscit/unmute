# syntax=docker.io/docker/dockerfile:1

FROM node:22-alpine AS dev

# Install required dependencies
RUN apk add --no-cache libc6-compat curl

# Set working directory
WORKDIR /app

# Install dependencies using pnpm
COPY package.json tsconfig*.json vite.config.ts pnpm-lock.yaml* .npmrc* ./
RUN corepack enable pnpm && pnpm i --frozen-lockfile

# Expose the port the dev server runs on
EXPOSE 3000

ENV NODE_ENV=development

HEALTHCHECK --start-period=15s \
    CMD curl --fail http://localhost:3000/ || exit 1

# The source code will be mounted as a volume, so no need to copy it here
# Default command to run the development server with hot reloading
CMD ["pnpm", "dev", "--host", "0.0.0.0", "--port", "3000"]
