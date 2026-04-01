import streamlit as st

st.title("토스 투자 정리")

uploaded_file = st.file_uploader("토스 캡처 업로드", type=["png","jpg","jpeg"])

if uploaded_file:
    st.image(uploaded_file)
    st.write("이미지 업로드 완료")
