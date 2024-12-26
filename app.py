import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import io

# Load the pre-trained model
model = load_model(r'C:\Users\lenovo\Brain O Vision\Project\Final\model.h5')  # Update this path if needed

# Define the categories (for example, three leaf diseases)
categories = ['early blight', 'late blight', 'healthy']  # Adjust based on your model categories

# Streamlit UI layout
st.title("Potato Leaf Disease Classification")

st.write("""
    Upload an image of a potato leaf, and the model will predict the disease type with confidence scores.
""")

# Dropdown for disease selection (optional, for reference)
disease_option = st.selectbox(
    "Select Leaf Disease Type",
    categories
)

# Image upload for prediction
uploaded_image = st.file_uploader("Upload Potato Leaf Image", type=['jpg', 'png', 'jpeg'])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Potato Leaf Image", width=100)

    # Convert image to the format suitable for the model
    img = image.load_img(uploaded_image, target_size=(224, 224))  # Adjust size based on your model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict using the model
    predictions = model.predict(img_array)

    # Get the confidence scores
    confidence_scores = predictions[0]  # Assuming the model outputs a vector of class probabilities

    # Display the confidence scores as a bar chart
    fig, ax = plt.subplots()
    ax.bar(categories, confidence_scores, color='skyblue')
    ax.set_xlabel("Leaf Disease Categories")
    ax.set_ylabel("Confidence Score")
    ax.set_title("Model Confidence Scores")
    st.pyplot(fig)

    # Display the most likely disease prediction
    predicted_class = categories[np.argmax(confidence_scores)]
    confidence = np.max(confidence_scores)
    st.write(f"**Prediction:** {predicted_class} with **confidence:** {confidence*100:.2f}%")
