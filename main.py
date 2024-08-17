import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import pickle
from PIL import Image
from sklearn.preprocessing import StandardScaler
import time

# Load the trained model and scaler
with open('rnn_lstm_model.pkl', 'rb') as f:
    model = pickle.load(f)
# Streamlit app title
st.title('Power System Stability Classification Demo')

def generate_random_features():
    # Generate random values for 20 features
    return np.random.uniform(0, 5000, 20).reshape(1, -1)

def update_plot(fig, features, iteration):
    # Clear previous plot
    plt.clf()
    ax = fig.add_subplot(111)
    
    features = features.flatten()
    bars = ax.bar(range(len(features)), features, color='royalblue')

    ax.set_title(f'Feature Values ')
    ax.set_xlabel('Features')
    ax.set_ylabel('Values')
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels([f'Feature {i+1}' for i in range(len(features))], rotation=-45)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom')  # va: vertical alignment

    return fig

# Create placeholders for results and the plot
result_placeholder = st.empty()
plot_placeholder = st.empty()
unstable_image = Image.open('unstable.jpeg') 
unstable_image_2=Image.open('unstable_2.jpeg') 
image_placeholder = st.empty()
# Initialize plot
fig = plt.figure(figsize=(10, 6))
plot_placeholder.pyplot(fig)

# Initialize iteration counter and stability status
iteration = 0
unstable_detected = False
col1, col2 = st.columns([1, 1])

# Create a column layout for displaying the plot and images side by side
col_plot, col_images = st.columns([2, 1])

while not unstable_detected:
    iteration += 1
    
    # Generate random features
    input_features = generate_random_features()

    # Standardize the features
    input_features = input_features.reshape(-1, input_features.shape[-1])
#     input_features = scaler.transform(input_features)
    input_features = input_features.reshape(-1, 1, input_features.shape[-1])

    # Make prediction
    try:
        y_pred_cat = model.predict(input_features)
        y_pred = np.argmax(y_pred_cat, axis=1)
    except Exception as e:
        result_placeholder.error(f"An error occurred: {str(e)}")
        st.stop()

    # Update the result display with color coding
    result_placeholder.subheader('Prediction')
    if y_pred[0] == 1:  # Stable
        result_placeholder.markdown('<h3 style="color: green;">The model predicts: Stable</h3>', unsafe_allow_html=True)
        image_placeholder.empty()
        unstable_detected = False
        
    else:  # Unstable
        result_placeholder.markdown('<h3 style="color: red;">The model predicts: Unstable</h3>', unsafe_allow_html=True)
#         image_placeholder.image(unstable_image, use_column_width=True)
#         image_placeholder.image(unstable_image_2, use_column_width=True)
        with col_images:
            st.image(unstable_image, use_column_width=True, width=500)
            st.image(unstable_image_2, use_column_width=True, width=500)
        unstable_detected = True

#     result_placeholder.subheader('Prediction Probability')
#     result_placeholder.write(f'Probability of stability: {y_pred_cat[0][1]:.2f}')
#     result_placeholder.write(f'Probability of instability: {y_pred_cat[0][0]:.2f}')

    # Update the interactive plot
    fig = update_plot(fig, input_features.reshape(-1, 20), iteration)
    plot_placeholder.pyplot(fig)

    # Wait for a short period before updating again
    time.sleep(2)  # Adjust the delay as needed

# Final message when instability is detected
st.write('The prediction has turned unstable. Stopping updates.')
