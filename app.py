import streamlit as st
import ml_utils

st.write('Hello world!')

options = st.multiselect(
    'What are your favorite colors',
    [ml_utils.get_available_models()]
)

st.write('You selected:', options)
