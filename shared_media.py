"""Shared-storage pyplot helper for multi-replica Streamlit deployments.

Saves matplotlib figures to a shared filesystem and displays them via
nginx's /media/ static location instead of Streamlit's in-memory media cache.
"""
import os
import hashlib
from io import BytesIO
import streamlit as st
import streamlit.components.v1 as components

MEDIA_DIR = os.environ.get("SHARED_MEDIA_DIR", "/shared/media")
MEDIA_URL = os.environ.get("SHARED_MEDIA_URL", "/media")


def shared_pyplot(fig, **kwargs):
    """Drop-in replacement for st.pyplot(fig) that writes to shared storage."""
    os.makedirs(MEDIA_DIR, exist_ok=True)
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    img_bytes = buf.getvalue()
    name = hashlib.sha256(img_bytes).hexdigest()[:16] + ".png"
    path = os.path.join(MEDIA_DIR, name)
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(img_bytes)
    url = f"{MEDIA_URL}/{name}"
    components.html(f'<img src="{url}" style="width:100%">', height=500, scrolling=True)
