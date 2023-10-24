import streamlit as st

# Define possible modelsd in a dict
# Format of the dict: model name -> model code

MODEL = {
    "model": "MOBO",
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
            inputs['method'] = st.selectbox('method', ['HV', 'EHVI'])
            inputs['normalize'] = st.selectbox('normalize',[None, 'StandardScaler', 'MinMaxScaler'])
            inputs['num'] = st.number_input('number', 1, 20, 1)
            inputs['kernel'] = st.selectbox('kernel', ['rbf', 'DotProduct + WhiteKernel'])
            # inputs['n dim'] = st.number_input('variable dim', 1, 20, 1)
            # inputs['size pop'] = st.number_input('size pop', 1, 500, 50)
            # inputs['max iter'] = st.number_input('max iter', 1, 10000, 200)
            # inputs['prob mut'] = st.slider('prob mut', 0.0, 1.0, 0.001)
            # inputs['F'] = st.slider('F', 0.0, 1.0, 0.5)

            # inputs['lb']  = st.number_input('lb',  )  # 跟据虚拟样本空间的最大值和最小值进行调整
            # inputs['ub'] = st.number_input('ub', )
    return inputs,col2
        
# To test the alg independent of the app or template, just run 
# `streamlit run alg.py` from within this folder.
if __name__ == "__main__":
    show()
    
    