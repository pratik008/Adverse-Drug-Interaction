import streamlit as st

st.title("Drug-Drug Interaction App")

drug1 = st.selectbox(label='Drug2', options=['Asperin', 'Advil', 'Tylenol', 'Unknown', 'my drug'])

drug2 = st.selectbox(label='Drug1', options=['Asperin', 'Acetomophin', 'Random', 'Unknown'])
st.write("My App Part 2")

st.write("%s+%s=%s"% (drug1,drug2,"Problem unknown"))



import tensorflow
tensorflow.__version__

