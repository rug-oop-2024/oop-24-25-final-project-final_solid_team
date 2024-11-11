import streamlit as st
    
def write_helper_text(text: str) -> None:
    """Inputs:
    text: str
    Writes text as helper text"""
    st.write(f'<p style="color: #888;">{text}</p>', unsafe_allow_html=True)
