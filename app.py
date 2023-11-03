import tensorflow as tf
from tensorflow import keras
import numpy as np
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

app = Flask(__name__, template_folder="templates")


# Step 1: Load the MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Step 2: Preprocess the data
train_images, test_images = train_images / 255.0, test_images / 255.0

# Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Step 4: Build a simple neural network model
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

model.save("my_model.h5")

# Load the trained model
model = keras.models.load_model('my_model.h5')

@app.route("/")
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/classify", methods=["POST"])
def classify():
    if request.method == "POST":
        # Rest of the code for image classification
        #We check if the request method is POST, which indicates that a form was submitted.
        # Get the uploaded file
        uploaded_file = request.files["file"]
        #We retrieve the uploaded file from the request using request.files["file"]. The "file" key corresponds to the name of the file input field in the HTML form.
        if uploaded_file.filename != "":  #We check if a file was actually uploaded by verifying that the filename attribute of the uploaded_file object is not empty.
            # Load and preprocess the uploaded image
            from PIL import Image

            # Inside the classify route
            img = Image.open(uploaded_file)
            img = img.resize((28, 28))
            img_array = np.array(img) / 255.0       
            # Make a prediction
            prediction = model.predict(np.array([img]))
            predicted_digit = np.argmax(prediction)

            return f"Predicted digit: {predicted_digit}"

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)