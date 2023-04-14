import streamlit as st


MODEL = {
    "model": "LassoRegressor",
}

def show():
    """Shows the components for the template and returns user inputs as dict."""
    
    
    inputs = {}  # dict to store all user inputs until return
    
    # template later.
    inputs["model"] = MODEL["model"]
    
    st.info('TO SOLVE **FEATURE IMPOTRANCE RANK**')
    
    col1, col2 = st.columns([2,2])
    with col1:
        with st.expander("Hyper Parameter"):

            random_state = st.checkbox('random state 1024',True)
            if random_state:
                inputs['random state'] = 1024
            else:
                inputs['random state'] = None

    return inputs,col2

if __name__ == "__main__":
    show()
    
    