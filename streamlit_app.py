import streamlit as st

st.title("Drug-Drug Interaction App")

drug1 = st.selectbox(label='Drug2', options=['Asperin', 'Advil', 'Tylenol', 'Unknown', 'my drug'])

drug2 = st.selectbox(label='Drug1', options=['Asperin', 'Acetomophin', 'Random', 'Unknown'])
st.write("My App Part 2")

st.write("%s+%s=%s"% (drug1,drug2,"Problem unknown"))

st.markdown("""
<div style='background-color: red; border: 1px solid black; color: black;height:100px;width:100px; text-align: center;'>
This is a box with red background
</div>
""", unsafe_allow_html=True)


st.markdown("""
<html>
<head>
<script type="text/javascript" language="javascript" src="jsme/jsme.nocache.js"></script>
<title>JME Example</title>
</head>

<body>
<div code="JME.class" name="JME" archive="JME.jar" width="360" height="315" id="JME">
You have to enable JavaScript in your browser to use JSME! </div>
</body>
</html>
""", unsafe_allow_html=True)



import tensorflow
tensorflow.__version__

