import streamlit as st

st.title("Drug-Drug Interaction App")

drug1 = st.selectbox(label='Drug1', options=['Asperin', 'Advil', 'Tylenol'])

drug2 = st.selectbox(label='Drug2', options=['Asperin', 'Advil', 'Tylenol'])

st.write("%s+%s=%s"% (drug1,drug2,"Problem unknown"))

import tensorflow
tensorflow.__version__
