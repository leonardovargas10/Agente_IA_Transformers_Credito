"""Frontend Streamlit do assistente.

Inicie o backend antes:
    uvicorn app_api:app --host 127.0.0.1 --port 8000
Depois execute:
    streamlit run 06_interface_web.py
"""
import json
import os

import requests
import streamlit as st

API_URL = os.getenv("ASSISTENTE_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Transformers e crédito", page_icon="💬", layout="wide")
st.title("Transformers e risco de crédito PF")
st.caption("Chat didático • Qwen 3B • ChromaDB • RAG com fontes")

st.session_state.setdefault("messages", [])

with st.sidebar:
    st.header("Configuração")
    model_mode = st.radio("Modelo", ["Base", "Ajustado"], index=0)
    use_rag = st.checkbox("Usar RAG", value=True)
    use_mmr = st.checkbox("Reduzir redundância com MMR", value=True)
    k = st.slider("Quantidade de trechos", min_value=1, max_value=8, value=4)
    if st.button("Nova conversa"):
        st.session_state.messages = []
        st.rerun()
    if st.button("Verificar backend"):
        try:
            response = requests.get(f"{API_URL}/health", timeout=10)
            response.raise_for_status()
            st.success("Backend disponível")
            st.json(response.json())
        except requests.RequestException as error:
            st.error(f"Backend indisponível: {error}")
    st.download_button(
        "Exportar conversa",
        data=json.dumps(st.session_state.messages, ensure_ascii=False, indent=2),
        file_name="conversa_transformers_credito.json",
        mime="application/json",
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for source in message.get("sources", []):
            label = f"{source.get('title', 'Fonte')} — p. {source.get('page', '?')}"
            with st.expander(label):
                st.write(source.get("text", ""))
                if source.get("section"):
                    st.caption(f"Seção: {source['section']}")
                if source.get("url"):
                    st.link_button("Abrir fonte", source["url"])

question = st.chat_input("Pergunte sobre Transformers ou aplicações em crédito")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    history = [
        {"role": item["role"], "content": item["content"]}
        for item in st.session_state.messages[:-1][-6:]
    ]
    payload = {
        "query": question,
        "history": history,
        "use_rag": use_rag,
        "use_mmr": use_mmr,
        "model_mode": model_mode,
        "k": k,
    }

    with st.chat_message("assistant"):
        try:
            with st.spinner("Consultando o backend..."):
                response = requests.post(f"{API_URL}/chat", json=payload, timeout=300)
            if response.status_code >= 400:
                detail = response.json().get("detail", response.text)
                raise RuntimeError(detail)
            result = response.json()
            st.markdown(result["answer"])
            for source in result.get("sources", []):
                label = f"{source.get('title', 'Fonte')} — p. {source.get('page', '?')}"
                with st.expander(label):
                    st.write(source.get("text", ""))
                    if source.get("section"):
                        st.caption(f"Seção: {source['section']}")
                    if source.get("url"):
                        st.link_button("Abrir fonte", source["url"])
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result.get("sources", []),
            })
        except (requests.RequestException, RuntimeError, ValueError) as error:
            mensagem = f"Não foi possível concluir a consulta: {error}"
            st.error(mensagem)
            st.session_state.messages.append({"role": "assistant", "content": mensagem, "sources": []})
