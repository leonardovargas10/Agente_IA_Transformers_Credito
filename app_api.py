"""Backend FastAPI do assistente didático de Transformers e crédito.

Executar:
    uvicorn app_api:app --host 127.0.0.1 --port 8000

O backend não carrega o Qwen durante a importação. O modelo é carregado na
primeira consulta para permitir testar a API e a coleção separadamente.
"""
from pathlib import Path
from threading import Lock
from typing import Literal

import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from rag import (
    COLECAO,
    ENCODER,
    ROOT,
    carregar_llm,
    gerar,
    montar_mensagens,
    recuperar,
)

app = FastAPI(
    title="Assistente didático de Transformers e crédito",
    version="0.1.0",
    description="API local com Qwen 3B, ChromaDB e RAG sobre documentos selecionados.",
)

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)

class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    history: list[Message] = Field(default_factory=list)
    use_rag: bool = True
    use_mmr: bool = True
    model_mode: Literal["Base", "Ajustado"] = "Ajustado"
    k: int = Field(default=4, ge=1, le=8)

class Source(BaseModel):
    source_id: str
    title: str = "Fonte sem título"
    page: int | str = "?"
    section: str = ""
    url: str = ""
    text: str

class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    model: str
    model_mode: str
    rag_enabled: bool

app_state = {
    "collection": None,
    "encoder": None,
    "models": {},
    "locks": {},
}

def get_resources():
    if app_state["collection"] is None:
        client = chromadb.PersistentClient(path=str(ROOT / "banco_chroma"))
        try:
            app_state["collection"] = client.get_collection(COLECAO)
        except Exception as error:
            raise RuntimeError(
                "Coleção Chroma ausente. Execute os notebooks 01 e 02 antes da API."
            ) from error
    if app_state["encoder"] is None:
        app_state["encoder"] = SentenceTransformer(ENCODER, device="cpu")
    return app_state["collection"], app_state["encoder"]

def get_model(mode: str):
    loaded_modes = set(app_state["models"])
    if loaded_modes and mode not in loaded_modes:
        raise RuntimeError("A API já carregou outro modo. Reinicie o backend para trocar entre Base e Ajustado.")
    if mode not in app_state["models"]:
        adapter = ROOT / "adaptador_pf" if mode == "Ajustado" else None
        if adapter is not None and not adapter.exists():
            raise RuntimeError(
                "O adaptador ainda não existe. Execute o notebook 04 ou utilize o modo Base."
            )
        model, tokenizer = carregar_llm(adapter)
        app_state["models"][mode] = (model, tokenizer)
        app_state["locks"][mode] = Lock()
    return app_state["models"][mode], app_state["locks"][mode]

def serialize_source(item: dict) -> Source:
    return Source(
        source_id=str(item.get("source_id", "desconhecida")),
        title=str(item.get("title", "Fonte sem título")),
        page=item.get("page", "?"),
        section=str(item.get("section", "")),
        url=str(item.get("url", "")),
        text=str(item.get("texto", "")),
    )

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "encoder": ENCODER,
        "collection": COLECAO,
        "model_loaded": list(app_state["models"].keys()),
    }

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        sources = []
        if request.use_rag:
            collection, encoder = get_resources()
            sources = recuperar(
                collection, encoder, request.query,
                usar_mmr=request.use_mmr, k=request.k, fetch_k=max(request.k * 3, 8),
            )
            if not sources:
                raise RuntimeError("A coleção não retornou evidências para a pergunta.")

        (model, tokenizer), lock = get_model(request.model_mode)
        history = [message.model_dump() for message in request.history[-6:]]
        messages = montar_mensagens(request.query, sources, history)
        with lock:
            answer = gerar(model, tokenizer, messages)

        return ChatResponse(
            answer=answer,
            sources=[serialize_source(item) for item in sources],
            model="Qwen/Qwen2.5-3B-Instruct",
            model_mode=request.model_mode,
            rag_enabled=request.use_rag,
        )
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Falha na consulta: {error}") from error
