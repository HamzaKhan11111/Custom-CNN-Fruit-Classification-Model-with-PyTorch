# Step 1: Select a base Python environment
FROM python:3.9-slim AS base

# Step 2: Set the working directory
WORKDIR /app

# Step 3: Copy only the requirements.txt file to the working directory
COPY requirements.txt .

# Step 4: Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy all content from the repository into the working directory
COPY . .

# Step 6: Train the model (replace this command with your actual model training command)
RUN python main.py

# Step 7: Create a Flask endpoint for the trained model
COPY model.pth /app
COPY app.py /app
EXPOSE 5000

# Step 8: Set the command to start the Flask app
CMD ["flask", "run", "--host=0.0.0.0"]

# Step 9: Optionally, you can add a stage for final image
FROM base AS final

# Step 10: Expose the port the Flask app will run on
EXPOSE 5000

# Step 11: Set the command to start the Flask app
CMD ["python", "app.py"]
