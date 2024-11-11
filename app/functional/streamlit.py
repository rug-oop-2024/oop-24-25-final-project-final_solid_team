import streamlit as st


def is_active(environment: str, condition: bool) -> bool:
    """Method to determine if sessions state was active"""
    # Add identifier to the session state
    if environment not in st.session_state:
        st.session_state[environment] = False

    # If the condition is true, start to make this section active
    if condition:
        st.session_state[environment] = True

    # Return wether the state has been made active in the past
    if st.session_state[environment]:
        return True
    else:
        return False
