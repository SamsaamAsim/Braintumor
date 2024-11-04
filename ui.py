import streamlit as st
from PIL import Image
import numpy as np
import cv2
from model import preprocess_image_from_array, load_model

# Load the trained model and PCA transformer
svm_classifier, pca_transformer = load_model()

# Function to predict if an image has a brain tumor or not
def classify_image(image, model, pca):
    try:
        # Preprocess the image
        feature_vector = preprocess_image_from_array(image)
        
        # Apply PCA transformation
        feature_vector_pca = pca.transform([feature_vector])

        # Run the classification model
        prediction = model.predict(feature_vector_pca)
        confidence = np.max(model.predict_proba(feature_vector_pca))  # Get the confidence level
        return prediction[0], confidence
    except Exception as e:
        st.error(f"Error during classification: {e}")
        return None, None

# Set the title of the Streamlit app
st.header('Brain Tumor Detection')

# Center-align the upload button
st.markdown(
    """
    <style>
    .centered {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 70vh;
    }
    .centered-button {
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display the upload button in the center
st.markdown('<div class="centered"><div class="centered-button">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
st.markdown('</div></div>', unsafe_allow_html=True)

if uploaded_file is not None:
    # Read the uploaded image file
    image = Image.open(uploaded_file).convert('RGB')  # Convert to RGB if not already
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Add a button to trigger prediction
    if st.button('Predict'):
        # Run the classification model
        prediction, confidence = classify_image(image, svm_classifier, pca_transformer)

        # Change page color based on prediction
        if prediction == 'no_tumor':
            st.markdown(
                """
                <style>
                body {
                    background-color: green !important;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <style>
                body {
                    background-color: red !important;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

        # Display the prediction result
        if prediction is not None:
            st.write(f"Prediction: {prediction}")
            st.write(f"Confidence Level: {confidence:.2f}")  # Display confidence level
