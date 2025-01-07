import streamlit as st
import requests

st.title("Smart City Waste Management")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Send image to API (use your actual API endpoint here)
    try:
        response = requests.post(
            'http://localhost:5000/predict', files={'file': uploaded_file.getvalue()}
        )

        if response.status_code == 200:
            prediction = response.json()['prediction']
            st.write(f"Predicted Label: {prediction}")
        else:
            st.error(f"Failed to get prediction. Status Code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred: {e}")
