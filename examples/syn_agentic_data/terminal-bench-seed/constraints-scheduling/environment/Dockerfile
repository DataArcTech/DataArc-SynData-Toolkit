# Terminal-Bench Canary String, DO NOT REMOVE:
FROM ubuntu:24.04
WORKDIR /app

# Install Python and basic tools for calendar processing
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
 && rm -rf /var/lib/apt/lists/*

# Copy input calendar files
COPY inputs/ /app/
