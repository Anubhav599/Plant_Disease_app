# This sets up the container with Python 3.10 installed.
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/root

# This tells Docker to listen on port 8501 (Streamlit default).
EXPOSE 8501

# This sets the /app directory as the working directory for any RUN, CMD, ENTRYPOINT, or COPY instructions that follow.
WORKDIR /app

# This copies everything in your current directory to the /app directory in the container.
COPY . /app

# This runs pip install for all the packages listed in your requirements.txt file.
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# This command creates a .streamlit directory in the home directory of the container.
RUN mkdir -p ~/.streamlit

# Create streamlit config file for Render deployment
RUN echo "[global]\n\
showWarningOnDirectExecution = false\n\
logLevel = 'error'\n\
\n\
[logger]\n\
level = 'error'\n\
\n\
[client]\n\
showErrorDetails = false\n\
\n\
[server]\n\
headless = true\n\
port = 8501\n\
enableCORS = false\n\
enableXsrfProtection = false" > ~/.streamlit/config.toml

# This sets the default command for the container to run the app with Streamlit.
ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
