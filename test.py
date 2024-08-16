import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical

# Load the trained model
with open('rnn_lstm_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize the scaler (you should fit it with the training data used)
scaler = StandardScaler()

# Streamlit app title
st.title('Power System Stability Classification')

# Input fields
st.sidebar.header('User Input')

def user_input_features():
    # Create sliders for 20 features
    features = []
    for i in range(1, 21):
        feature = st.sidebar.slider(f'Feature {i}', 0.0, 10000.0, 10000.0)
        features.append(feature)

    # Convert the list of features to a NumPy array and reshape for the model
    return np.array(features).reshape(1, -1)
def plot_interactive_graph(features):
    # Create a DataFrame for Plotly
    df = pd.DataFrame({
        'Feature': [f'Feature {i}' for i in range(1, 21)],
        'Value': features.flatten()
    })
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['Feature'], y=df['Value'], marker_color='royalblue'))

    fig.update_layout(
        title='Feature Values',
        xaxis_title='Features',
        yaxis_title='Values',
        xaxis_tickangle=-45
    )

    st.plotly_chart(fig)

# Get user input
input_features = user_input_features()

# Standardize the features
input_features = input_features.reshape(-1, input_features.shape[-1])
# input_features = scaler.fit_transform(input_features)  # Assuming you fit scaler with the same data used for training
input_features = input_features.reshape(-1, 1, input_features.shape[-1])  # Reshape to 3D

# Make prediction
y_pred_cat = model.predict(input_features)
print(y_pred_cat)
y_pred = np.argmax(y_pred_cat, axis=1)

# Display results
st.subheader('Prediction')
st.write(f'The model predicts: {"Stable" if y_pred[0] == 1 else "Unstable"}')

st.subheader('Prediction Probability')
st.write(f'Probability of stability: {y_pred_cat[0][1]:.2f}')
st.write(f'Probability of instability: {y_pred_cat[0][0]:.2f}')
# Display the interactive plot
st.subheader('Feature Values Interactive Plot')
plot_interactive_graph(input_features.reshape(-1, 20))
