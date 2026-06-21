# ─────────────────────────────────────────────────────────────
# Dockerfile
# Base image: Python 3.9 slim
# Runs a Streamlit app for tourism package purchase prediction
# ─────────────────────────────────────────────────────────────

FROM python:3.9

# Set working directory
WORKDIR /app

# Copy all files into the container
COPY . .

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Create a non-root user (Hugging Face requirement)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app
COPY --chown=user . $HOME/app

# Start Streamlit app on port 8501
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.enableXsrfProtection=false"]
