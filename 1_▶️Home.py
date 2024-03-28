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

# --- SKILLS ---
st.write('\n')
st.subheader("Hard Skills")
st.write(
    """
- 👩‍💻 Programming: Python (Scikit-learn, PyTorch, TensorFlow), SQL e PySpark.
- 📊 Data Visulization: PowerBi, MS Excel, Plotly, Seaborn
- 🗄️ Databases: Postgres, MongoDB, MySQL
"""
)

# --- SKILLS ---

st.write('\n')
st.subheader("Projetos Reais")
st.write(
    """
**✅ Sistema de Predição de Manutenção de Equipamentos**
- Desenvolvimento de um modelo de classificação para prever a necessidade de manutenção em máquinas pesadas de uma empresa do setor industrial e de construção.
- Utilização de dados de série temporal das horas de funcionamento das máquinas e informações detalhadas sobre o equipamento para criar um modelo preditivo robusto.
- Implementação de algoritmos de machine learning, incluindo XGBoost, para prever horas de funcionamento futuras e identificar máquinas com potencial necessidade de manutenção.
- Integração de dados climáticos obtidos por API para enriquecer o modelo e considerar fatores ambientais na predição de manutenção.
- Desenvolvimento de um dashboard interativo utilizando Streamlit para visualização das previsões de horas de trabalho, calendário de manutenção e informações gerais sobre o equipamento.
"""
)

st.write('\n')
st.write(
    """
     **✅ Sistema de Recomendação Content-Based** 

- Elaboração de sistema de recomendação content-based para aplicativos de autoatendimento em estabelecimentos gastronômicos;

- Análise de dados não estruturados aplicando técnicas de Processamento de Linguagem Natural, Text Mining e Machine Learning;

- Suporte ao time de desenvolvimento com a documentação técnica do projeto;

- Principal resultado alcançado: Desenvolvimento de API para sistema de recomendação content based atuando em diferentes franquias, utilizando histórico de compra do usuário e recomendando itens de acordo com os perfis semelhantes ao do usuário.
"""
)
st.write('[Post LinkedIn](https://www.linkedin.com/posts/aurelio-guilherme-silva_python-recommendationabrsystem-contentbased-activity-7108935048350842880-vV1X?utm_source=share&utm_medium=member_desktop)')
st.write('\n')

st.write(
    '''
    **✅ Detecção de Presença em Ambientes Monitorados - IoT**
- Desenvolvimento de um modelo preditivo de classificação utilizando Python e framework CRISP-DM para detecção de pessoas em um ambiente monitorado;
- Feature engineering para melhoria da acurácia do modelo preditivo;
- Principal resultado alcançado: Elaboração de pipeline de treino e monitoramento de modelos de machine learning, resultando em modelos de classificação com capacidade de predizer presença de pessoas em um ambiente monitorado. Alcançando consistentemente uma acurácia superior a 80%, contribuindo para a otimização do consumo energético dos sistemas de ar-condicionado.
'''
)







