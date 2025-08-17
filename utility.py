import streamlit as st
import random
import hmac

"""
This file contains the common components used in the Streamlit App.
This includes the sidebar, the title, the footer, and the password check.
"""

def check_password():
    """Returns `True` if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            # Store the OpenAI API key from secrets for use in the app
            st.session_state["validated_openai_key"] = st.secrets["openai_api_key"]
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.markdown("## 🔐 Login Required")
    st.markdown("Please enter your password to access the Case Summary Recommendation System.")
    
    st.text_input(
        "Password", 
        type="password", 
        on_change=password_entered, 
        key="password",
        placeholder="Enter system password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False