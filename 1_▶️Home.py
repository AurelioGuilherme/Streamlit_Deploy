from pathlib import Path
import streamlit as st
from PIL import Image


# --- Path Settings---
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()

css_file = current_dir / 'styles' / 'main.css'
resume_file = current_dir / 'assets' / 'CV.pdf'
profile_pic = current_dir / 'assets' / 'profile-pic.png'

# --- General Settings ---

PAGE_TITLE = 'Meu Porfólio de Projetos 🎖️'
PAGE_ICON = "🎖️"
NAME = 'Meu Porfólio'
DESCRIPTION = '''Bem-vindo ao meu portfólio de projetos em ciência de dados! Aqui você encontrará uma coleção de projetos nos quais explorei e apliquei técnicas de machine learning, deep learning e engenharia de dados para resolver problemas do mundo real.

Navegue pelos meus projetos para explorar exemplos de minha abordagem analítica e criativa para resolver problemas em diversas áreas, desde predição de séries temporais até reconhecimento de padrões em imagens.

Se você está interessado em colaborar ou discutir oportunidades de trabalho, não hesite em entrar em contato. Estou sempre aberto a novas ideias e desafios!

'''

EMAIL = 'aurelio_gss@hotmail.com'
SOCIAL_MEDIA = {"LinkedIn": "https://www.linkedin.com/in/aurelio-guilherme-silva/",
                "GitHub": "https://github.com/AurelioGuilherme",
                'Instagram': "https://www.instagram.com/aurelio_gss/",
                "Hugging Face": "https://huggingface.co/Aurelio-Guilherme"}


st.set_page_config(page_title=PAGE_TITLE,page_icon=PAGE_ICON)

st.title("Aurélio Guilherme")

# --- LOAD CSS, PDF & PROFIL PIC ---
with open(css_file) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
with open(resume_file, "rb") as pdf_file:
    PDFbyte = pdf_file.read()
profile_pic = Image.open(profile_pic)

# --- HERO SECTION ---
col1, col2 = st.columns(2, gap="small")
with col1:
    st.image(profile_pic, width=230)

with col2:
    st.write(DESCRIPTION)
    st.download_button(
        label=" 📄 Download Resume",
        data=PDFbyte,
        file_name=resume_file.name,
        mime="application/octet-stream",
    )
    st.write("📫", EMAIL)


# --- SOCIAL LINKS ---
st.write('\n')
cols = st.columns(len(SOCIAL_MEDIA))
for index, (platform, link) in enumerate(SOCIAL_MEDIA.items()):
    cols[index].write(f"[{platform}]({link})")


