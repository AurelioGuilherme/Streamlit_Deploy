import streamlit as st
import pandas as pd
import plotly.express as px

def main():
    # Carregar o DataFrame (substitua esta linha pelo seu próprio DataFrame)
    df = pd.read_excel("cardio.xlsx")

    # Exibir o DataFrame
    st.write("## DataFrame:")
    st.write(df)

    # Opções para selecionar coluna
    coluna_selecionada = st.selectbox("Selecione uma coluna:", df.columns)

    # Gerar estatísticas básicas para a coluna selecionada
    st.write("## Estatísticas básicas para a coluna selecionada:")
    st.write(df[coluna_selecionada].describe())

    # Opções para selecionar o tipo de gráfico
    tipo_grafico = st.selectbox("Selecione o tipo de gráfico:",
                                ["Pizza", "Barras", "Linhas"])

    # Gerar o gráfico conforme o tipo selecionado
    if tipo_grafico == "Pizza":
        fig = px.pie(df, names=df[coluna_selecionada])
    elif tipo_grafico == "Barras":
        fig = px.bar(df, x=df.index, y=coluna_selecionada)
    elif tipo_grafico == "Linhas":
        fig = px.line(df, x=df.index, y=coluna_selecionada)

    # Exibir o gráfico
    st.write("## Gráfico:")
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
