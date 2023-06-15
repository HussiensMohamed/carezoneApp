from flask import Flask, jsonify, request, send_file, render_template
import numpy as np
import cv2
import tensorflow as tf
import uuid
import base64

app = Flask(__name__)
model = tf.keras.models.load_model("BrainTumor10.h5")
image_size = 150
# Define a custom Jinja2 filter for base64 encoding
def b64encode(data):
    return base64.b64encode(data).decode('utf-8')

# Register the custom Jinja2 filter
app.jinja_env.filters['b64encode'] = b64encode


def draw_circle(img, prediction):
    # Get the class with the highest probability
    class_idx = np.argmax(prediction[0])

    # Define the class labels
    classes = ["glioma Tumour", "meningioma Tumour", "No Tumour", "Pituitary Tumour"]

    # Get the predicted class label
    predicted_class = classes[class_idx]

    # Draw a circle around the tumor
    if predicted_class != "No Tumour":
        # Convert the prediction to a probability
        prob = prediction[0][class_idx]

        # Scale the radius of the circle based on the probability
        radius = int(prob * 20)

        # Find the edges in the image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Find the contours of the edges
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the contour with the largest area
        max_area_contour = max(contours, key=cv2.contourArea)

        # Find the center of the contour using moments
        M = cv2.moments(max_area_contour)
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])

        # Draw the circle on the input image
        cv2.circle(img, (center_x, center_y), radius, (0, 0, 255), 2)

    return img, predicted_class

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the request
    image = request.files['image'].read()

    # Convert the image to a NumPy array
    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (image_size, image_size))
    img_array = np.array(img)
    img_array = img_array.reshape(1, image_size, image_size, 3)
    

    # Make a prediction using your model
    prediction = model.predict(img_array)

    # Draw a circle around the detected tumor
    img_with_circle, predicted_class = draw_circle(img, prediction)



    # Encode the image as a byte array
    _, buffer = cv2.imencode('.jpg', img_with_circle)

    # Render the result.html template with the predicted class and image data
    return render_template('result.html', predicted_class=predicted_class, image_data=buffer.tobytes())

@app.route('/image/<path:path>')
def get_image(path):
    # Return the image file
    return send_file(path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)