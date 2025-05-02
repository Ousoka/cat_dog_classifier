# import streamlit as st
# import tensorflow as tf
# from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
# import numpy as np
# from PIL import Image

# # Custom layer for loading a TensorFlow SavedModel in Keras 3.x
# def load_classification_model():
#     try:
#         # Use TFSMLayer to load the model
#         model = tf.keras.layers.TFSMLayer("efficientnetv2_catdog_model", call_endpoint='serving_default')
#         return model
#     except Exception as e:
#         st.error(f"❌ Erreur lors du chargement du modèle: {e}")
#         return None

# # Classes (adjust if needed based on training order)
# class_names = ['Chat', 'Chien']

# # UI
# st.title("🧠 Reconnaissance : Chat vs Chien")
# st.write("Chargez une image et laissez l'IA prédire si c'est un **Chat** ou un **Chien** 🐾")

# # Load model
# model = load_classification_model()

# if model:
#     uploaded_file = st.file_uploader("📷 Choisir une image...", type=["jpg", "jpeg", "png"])

#     if uploaded_file is not None:
#         image = Image.open(uploaded_file).convert("RGB")
#         st.image(image, caption="Image chargée", use_container_width=True)

#         if st.button("🔍 Lancer la prédiction"):
#             with st.spinner("Prédiction en cours..."):
#                 try:
#                     # Resize and preprocess
#                     img = image.resize((224, 224))
#                     img_array = np.array(img)
#                     img_preprocessed = preprocess_input(img_array)
#                     img_batch = np.expand_dims(img_preprocessed, axis=0)

#                     # Predict
#                     prediction = model.predict(img_batch)
#                     predicted_index = np.argmax(prediction)
#                     predicted_class = class_names[predicted_index]
#                     confidence = float(np.max(prediction)) * 100

#                     # Results
#                     st.success(f"✅ Cette image est un **{predicted_class}**")
#                     st.info(f"Confiance du modèle : {confidence:.2f}%")

#                     # Bar chart of prediction scores
#                     st.bar_chart({class_names[i]: float(prediction[0][i]) for i in range(len(class_names))})

#                 except Exception as e:
#                     st.error(f"❌ Erreur pendant la prédiction: {e}")
# else:
#     st.warning("Le modèle n'a pas pu être chargé. Veuillez vérifier le dossier 'efficientnetv2_catdog_model'.")



import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import numpy as np
from PIL import Image

# Function to load the model
def load_classification_model():
    try:
        # Load the SavedModel directly (ensure it's a SavedModel format)
        model = tf.saved_model.load("efficientnetv2_catdog_model")
        return model
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle: {e}")
        return None

# Classes in training order (adjust if needed)
class_names = ['Chat', 'Chien']

# Interface Streamlit
st.title("🧠 Reconnaissance : Chat vs Chien")
st.write("Chargez une image et laissez l'IA prédire si c'est un **Chat** ou un **Chien** 🐾")

# Load the model
model = load_classification_model()

if model:
    uploaded_file = st.file_uploader("📷 Choisir une image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Image chargée", use_container_width=True)

        if st.button("🔍 Lancer la prédiction"):
            with st.spinner("Prédiction en cours..."):
                try:
                    # Resize and preprocess the image
                    img = image.resize((224, 224))
                    img_array = np.array(img)
                    
                    # Ensure the image is float32
                    img_array = img_array.astype('float32')  # Convert to float32
                    img_preprocessed = preprocess_input(img_array)
                    img_batch = np.expand_dims(img_preprocessed, axis=0)

                    # Prediction using the model
                    prediction_fn = model.signatures["serving_default"]  # This function should handle inference
                    prediction = prediction_fn(tf.convert_to_tensor(img_batch))

                    # Get the prediction values
                    output_tensor = prediction['output_0']  # Ensure to match the correct output name (may vary)
                    prediction_values = output_tensor.numpy()

                    # Get the predicted class
                    predicted_index = np.argmax(prediction_values)
                    predicted_class = class_names[predicted_index]
                    confidence = float(np.max(prediction_values)) * 100

                    # Display results
                    st.success(f"✅ Cette image est un **{predicted_class}**")
                    st.info(f"Confiance du modèle : {confidence:.2f}%")

                    # Bar chart of prediction scores
                    st.bar_chart({class_names[i]: float(prediction_values[0][i]) for i in range(len(class_names))})

                except Exception as e:
                    st.error(f"❌ Erreur pendant la prédiction: {e}")
else:
    st.warning("Le modèle n'a pas pu être chargé. Veuillez vérifier le dossier 'efficientnetv2_catdog_model'.")
