# Use the official Python 3.10.12 image
FROM python:3.10.12

# Install Flask, gunicorn, OpenCV, and webcolors
RUN pip install Flask gunicorn opencv-python webcolors

# Copy the source code into the /app directory
COPY src/ /app

# Set the working directory to /app
WORKDIR /app

# Set the environment variable for the port
ENV PORT 8080

# Start the application using gunicorn
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app
