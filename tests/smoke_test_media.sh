#!/bin/bash
# Smoke test: verifies shared_pyplot writes to /shared/media and nginx serves it
set -e

NGINX_URL="http://localhost:8080"
MAX_WAIT=60

echo "Waiting for nginx to be healthy..."
elapsed=0
until curl -sf "$NGINX_URL/" > /dev/null 2>&1; do
  sleep 2
  elapsed=$((elapsed + 2))
  if [ $elapsed -ge $MAX_WAIT ]; then
    echo "FAIL: nginx not healthy after ${MAX_WAIT}s"
    exit 1
  fi
done
echo "nginx is up after ${elapsed}s"

# Write a test image to shared media via the main container
echo "Writing test image to shared media..."
docker compose -f docker-compose.ci.yml exec -T main python -c "
import matplotlib.pyplot as plt
from shared_media import shared_pyplot, MEDIA_DIR
import os
import unittest.mock as mock

fig, ax = plt.subplots()
ax.plot([1,2,3], [4,5,6])

with mock.patch('streamlit.components.v1.html'):
    shared_pyplot(fig)
plt.close(fig)

files = os.listdir(MEDIA_DIR)
assert len(files) >= 1, f'No files in {MEDIA_DIR}'
print(files[0])
"

# Get the filename that was written
MEDIA_FILE=$(docker compose -f docker-compose.ci.yml exec -T main python -c "
import os
files = [f for f in os.listdir('/shared/media') if f.endswith('.png')]
print(files[0])
")
MEDIA_FILE=$(echo "$MEDIA_FILE" | tr -d '\r\n')

echo "Testing nginx serves /media/${MEDIA_FILE}..."
HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" "$NGINX_URL/media/${MEDIA_FILE}")

if [ "$HTTP_CODE" = "200" ]; then
  echo "PASS: nginx serves /media/${MEDIA_FILE} with HTTP 200"
else
  echo "FAIL: nginx returned HTTP ${HTTP_CODE} for /media/${MEDIA_FILE}"
  exit 1
fi

# Verify content-type is an image
CONTENT_TYPE=$(curl -sf -o /dev/null -w "%{content_type}" "$NGINX_URL/media/${MEDIA_FILE}")
if echo "$CONTENT_TYPE" | grep -q "image"; then
  echo "PASS: content-type is ${CONTENT_TYPE}"
else
  echo "FAIL: unexpected content-type: ${CONTENT_TYPE}"
  exit 1
fi

echo ""
echo "All smoke tests passed."
