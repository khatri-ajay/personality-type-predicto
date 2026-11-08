import streamlit as st

st.title("My first app")

name = st.text_input("What's your name?")
mood = st.slider("How are you feeling? (1-5)", 1, 5, 3)

if st.button("Submit"):
    st.success(f"Hi {name}! You picked {mood}.")