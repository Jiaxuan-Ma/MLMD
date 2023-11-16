import streamlit as st

# Define possible modelsd in a dict
# Format of the dict: model name -> model code

MODEL = {
    "model": "SupportVector",
}

# LightGBM can use -- categorical features -- as input directly. It doesn’t need to convert 
# to one-hot encoding, and is much faster than one-hot encoding (about 8x speed-up).

def show():
    """Shows the components for the template and returns user inputs as dict."""
    
    # `show()` is the only method required in this module. You can add any other code 
    # you like above or below. 
    
    inputs = {}  # dict to store all user inputs until return
    
    # with st.sidebar:
        
    # Render all template-specific sidebar components here. 

    # Use ## to denote sections. Common sections for training templates: 
    # Model, Input data, Preprocessing, Training, Visualizations
    # Store all user inputs in the `inputs` dict. This will be passed to the code
    # template later.
    inputs["model"] = MODEL["model"]
    
    # st.write("preprocessing")
    # inputs["Normalize"] = st.selectbox('normalize method', ['Z-Score Standardization','Min-Max Scale'])
    
    st.info('TO SOLVE **CLASSIFICATION**')
    
    # st.write('training')

    # st.write("No additional parameters")
    col1, col2 = st.columns([2,2])
    with col1:
        with st.expander("Hyper Parameter"):
            inputs['kernel'] = st.selectbox('kernel',['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'])
            inputs['C'] = st.number_input('C', 1, 20000, 1000)
            # inputs['min samples split'] = st.number_input('min samples split', 2, 1000, 2)
            
            random_state = st.checkbox('random state 42',True)
            if random_state:
                inputs['random state'] = 42
            else:
                inputs['random state'] = None
             # graph parameter
        with st.expander("Unbalanced Data"):
            inputs['unbalanced data'] = st.checkbox('unbalanced data', False)
            if inputs['unbalanced data']:
                inputs['min class weight'] = st.number_input('min class weight', 1, 1000, 1)
                inputs['max class weight'] = st.number_input('max class weight', 1, 1000, 1)
                inputs['class weight'] = {inputs['min class weight']:inputs['max class weight']}
            else:
                inputs['class weight'] = None
            auto_hyperparameters = st.checkbox('auto hyperparameters',False)
            if auto_hyperparameters:
                inputs['auto hyperparameters'] = True
                inputs['init points'] = st.number_input('init points',1, 100, 10)
                inputs['iteration number'] = st.number_input('iteration number',1, 500, 10)
            else:
                inputs['auto hyperparameters'] = False     

    return inputs,col2


# To test the alg independent of the app or template, just run 
# `streamlit run alg.py` from within this folder.
if __name__ == "__main__":
    show()
    
    