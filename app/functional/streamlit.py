import streamlit as st
from uuid import uuid4  # For returning garuanteed unique numbers

    

def active(condition: bool, unique_identifier: str) -> bool:
    # Add identifier to the session state
    if unique_identifier not in st.session_state:
        st.session_state[unique_identifier] = False
    
    # If the condition is true, start to make this section active
    if condition:
        st.session_state[unique_identifier] = True
    
    # Return wether the state has been made active in the past
    if st.session_state[unique_identifier]:
        return True
    else:
        return False
    