import streamlit as st

# Define possible modelsd in a dict
# Format of the dict: model name -> model code

MODEL = {
    "model": "LinearRegressor",
}

# LightGBM can use -- categorical features -- as input directly. It doesn’t need to convert 
# to one-hot encoding, and is much faster than one-hot encoding (about 8x speed-up).

def show():
    """Shows the components for the template and returns user inputs as dict."""
    
    # `show()` is the only method required in this module. You can add any other code 
    # you like above or below. 
    
    inputs = {}  # dict to store all user inputs until return
    
    inputs["model"] = MODEL["model"]
    
    
    st.info('TO SOLVE **FEATURE IMPOTRANCE RANK**')
    
    # st.write('training')

    # st.write("No additional parameters")
    col1, col2 = st.columns([2,2])
    with col1:
        with st.expander("Hyper Parameter"):
            # inputs['penalty'] = st.selectbox('penalty',('l2','l1','elasticnet','None'))
        
            # inputs['C'] = st.number_input('C', 0.01, 100.0, 1.0)
            # inputs['solver'] = st.selectbox('solver',('lbfgs','liblinear','newton-cg','newton-cholesky','sag','saga'))
            # inputs['max iter'] = st.number_input('max iter', 0, 100000, 100)
            # inputs['multi class'] = st.selectbox('multi class',('auto','ovr','multinomial'))
            # if inputs['penalty'] == 'elasticnet':
            #     inputs['l1 ratio'] = st.slider('l1 ratio',0.0, 1.0, 0.5) 
            # else:
            #     inputs['l1 ratio'] = None

            random_state = st.checkbox('random state 1024',True)
            if random_state:
                inputs['random state'] = 1024
            else:
                inputs['random state'] = None
       
        # with st.expander("Unbalanced Data"):
        #     inputs['unbalanced data'] = st.checkbox('unbalanced data', False)
        #     if inputs['unbalanced data']:
        #         inputs['class weight'] = st.selectbox('class weight',(None,'balanced'))
        #     else:
        #         inputs['class weight'] = None

    return inputs,col2


# To test the alg independent of the app or template, just run 
# `streamlit run alg.py` from within this folder.
if __name__ == "__main__":
    show()
    
    