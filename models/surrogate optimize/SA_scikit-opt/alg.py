import streamlit as st

# Define possible modelsd in a dict
# Format of the dict: model name -> model code

MODEL = {
    "model": "SA",
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
    
    
    st.info('TO SOLVE **OPT**')

    col1, col2 = st.columns([2,2])
    with col1:
        with st.expander("Hyper Parameter"):
            inputs['objective'] = st.selectbox('objective', ['max', 'min'])
            inputs['n dim'] = st.number_input('variable dim', 1, 20, 10)
            inputs['T max'] = st.number_input('T max', 1, 10, 1)
            inputs['q'] = st.number_input('q', 0.0, 2.0, 0.99)
            
            inputs['L']  = st.number_input('L', 0, 1000, 300)
            inputs['max stay counter'] = st.number_input('max stay counter', 0, 1000, 150)
            # inputs['lb']  = st.number_input('lb',  )  # 跟据虚拟样本空间的最大值和最小值进行调整
            # inputs['ub'] = st.number_input('ub', )
    return inputs,col2
        
# To test the alg independent of the app or template, just run 
# `streamlit run alg.py` from within this folder.
if __name__ == "__main__":
    show()
    
    