import streamlit as st
import openai
import os
from dotenv import load_dotenv

# 1. Carregar variáveis de ambiente
load_dotenv()

# É mais seguro usar variáveis de ambiente em produção/deploy
openai.api_key = os.getenv("OPENAI_API_KEY")

# ID do modelo refinado criado.
FINE_TUNED_MODEL_ID = "ft:gpt-3.5-turbo-0125:personal::BcPgmbYx"

# 4. Configurações da página Streamlit
st.set_page_config(
    page_title="Especialista em Cannabis Medicinal",
    page_icon="",
    layout="centered"
)

st.title("Olá, sou o Dr. Cannabis!")
st.markdown("Você tem dúvidas sobre cannabis medicinal? Me faça uma pergunta!")

# 5. Inicializar o histórico de chat no Session State
# O Streamlit Session State permite que os dados persistam entre as interações do usuário
if "messages" not in st.session_state:
    st.session_state.messages = []

# 6. Exibir mensagens anteriores do histórico de chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 7. Capturar entrada do usuário
if prompt := st.chat_input("Faça sua pergunta aqui..."):
    # Adicionar a mensagem do usuário ao histórico e exibir
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Chamar a API da OpenAI com o modelo fine-tuned
    with st.chat_message("assistant"):
        with st.spinner("Estou pensando..."):
            try:
                response = openai.chat.completions.create(
                    model=FINE_TUNED_MODEL_ID,
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages # Envia o histórico completo para contexto
                    ],
                    temperature=0.8,  # Ajuste para controlar a criatividade (0.0-1.0)
                    max_tokens=500   # Limite o comprimento da resposta
                )
                assistant_response = response.choices[0].message.content
                st.markdown(assistant_response)
                # Adicionar a resposta do assistente ao histórico
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})

            except openai.APIError as e:
                st.error(f"Erro ao interagir com o Dr. Cannabis: {e}")
                st.info("Verifique se você inseriu corretamente o contato do Dr. Cannabis.")
            except Exception as e:
                st.error(f"Erro inesperado: {e}")

# 8. Botão para limpar o histórico do chat
if st.button("Limpar Conversa"):
    st.session_state.messages = []
    st.rerun() # Reinicia a aplicação para limpar a tela