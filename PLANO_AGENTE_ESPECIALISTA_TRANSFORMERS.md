# Plano vigente — Agente de estudo sobre Transformers e crédito PF

Este arquivo é o contexto operacional do projeto. Deve permanecer curto, atualizado pelos artefatos existentes e suficiente para orientar novas alterações.

## Objetivo e limites

Construir uma aplicação local e didática para estudar Transformers, RAG, QLoRA e hipóteses de aplicação em risco de crédito PF. A aplicação explica conceitos, matemática, código PyTorch e desenho de experimentos; não é motor de aprovação, modelo de PD ou sistema de decisão. As respostas devem separar evidência documental, dedução didática e exemplo construído. Em propostas de crédito, explicitar unidade de análise, data de referência, janelas, alvo, maturação, leakage, baseline LightGBM, divisão temporal, calibração, estabilidade e impacto de política. Não usar dados reais ou confidenciais.

## Decisões fixas

- LLM única: Qwen/Qwen2.5-3B-Instruct.
- Quantização: 4 bits, NF4 quando suportado, FP16 validado na RTX 2060 SUPER de 8 GB.
- Encoder: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2, inicialmente em CPU.
- Banco: Chroma persistente, HNSW e space=cosine.
- Recuperação: busca exata versus ANN; MMR é alternativa avaliada, não melhoria presumida.
- Deep Learning: PyTorch. Boosting: LightGBM.
- Ajuste: QLoRA/PEFT após o baseline de RAG.
- Frontend: Streamlit. Backend: FastAPI. Conversas separadas do Chroma.
- Semente: RANDOM_STATE = 42, quando aplicável.

## Ordem vigente

00_ambiente_tokenizacao_e_llm.ipynb — ambiente, tokenização, template, shapes, inferência e memória.
01_documentos_extracao_e_chunks.ipynb — extração, catálogo, revisão, chunks e embeddings; gera dados_processados/.
02_banco_vetorial_hnsw_cosseno_e_mmr.ipynb — Chroma, HNSW/cosseno, busca exata, ANN e MMR; gera banco_chroma/.
03_rag_com_fontes_e_qwen.ipynb — recuperação, prompt, evidências e fontes.
04_fine_tuning_qlora.ipynb — dataset SFT, labels, máscara, passo didático e piloto; gera adaptador_pf/.
05_avaliacao_base_rag_e_ajuste.ipynb — quatro condições base/ajustada × sem/com RAG; gera JSONs para revisão.
06_interface_web.py + app_api.py — chat Streamlit separado da API FastAPI.

99_notebook_integrado_referencia.ipynb e materiais do curso são referências históricas. 02_preparar_banco.py é auxiliar; rag.py contém lógica operacional compartilhada e compreendida nos notebooks.

## Estado atual verificado

- Três PDFs públicos, 103 páginas extraídas e catálogo com URLs/hashes em documentos/catalogo.json.
- 990 chunks e embeddings normalizados em dados_processados/.
- Coleção Chroma reaberta em novo processo com 990 registros.
- HNSW em cosseno, comparação demonstrativa com busca exata e MMR testado.
- Interface inicial validada com o mecanismo de testes do Streamlit.
- dataset_sft.jsonl: piloto com 8 exemplos de treino, 2 de validação e 2 de teste; insuficiente para declarar especialização.
- Código de QLoRA e avaliação implementados, mas o ajuste completo ainda não foi executado.
- A geração completa do Qwen 3B ainda precisa ser registrada como validada.

Recall ANN@4 = 1 em uma consulta é apenas demonstração, não benchmark geral.

## Corpus inicial

Attention Is All You Need; Predicting Consumer Default; PRAGMA. Esses artigos não comprovam eficácia para crédito PF brasileiro. Fórmulas e tabelas extraídas de PDF exigem revisão visual. PDFs não alimentam automaticamente o fine-tuning.

## Contratos técnicos

Normalizar documentos e consultas com L2; validar dimensão, finitude e norma. Para vetores unitários, s_cos(q,x)=q·x e d_cos=1-s_cos; distância não é probabilidade. Registrar modelo, revisão, métrica, chunking, corpus e HNSW. IDs estáveis devem impedir duplicação e permitir atualização sem versões obsoletas. Comparar ANN com vizinhos exatos usando Recall_ANN@k e medir construção, armazenamento, RAM e latência p50/p95.

Fluxo RAG: pergunta → consulta → trechos → prompt → resposta → fontes. Começar com fluxo fixo; ferramentas e busca iterativa somente se uma falha avaliada justificar. Limitar rodadas, tokens e tempo. Perguntas sem suporte devem gerar limitação explícita. Histórico conversacional não é banco documental.

No QLoRA, inspecionar batch e passo (forward, loss, backward, acumulação e atualização) antes do trainer; explicar deslocamento causal, padding, ignore_index=-100 e tokens supervisionados. Separar treino, validação e teste por origem/família. Comparar base sem RAG, base com RAG, ajustada sem RAG e ajustada com RAG. Avaliar correção, matemática, fidelidade, citações, falta de evidência, utilidade, latência e memória. Loss menor isoladamente não comprova melhor resposta.

## Execução

O app_api.py expõe /health e /chat; o 06_interface_web.py é frontend Streamlit. Após os notebooks 01 e 02:

conda activate dsadlp1
Set-Location D:\Projetos_v2\Agente_IA_Transformers_Credito
uvicorn app_api:app --host 127.0.0.1 --port 8000
python -m streamlit run 06_interface_web.py --server.address 127.0.0.1

Fechar kernels e reiniciar a API ao alternar base/adaptador. Não carregar modelos alternativos automaticamente.

## Critério de entrega

Uma etapa só está concluída quando possui artefato reconstruível, execução limpa, inspeção dos intermediários, interpretação, limitações e atualização do README.md. Verificar idempotência, atualização sem resíduos, corpus vazio, pergunta sem suporte, limite de contexto, falha de carregamento, incompatibilidade de adaptador e isolamento de conversas. Não preencher resultados fictícios nem afirmar execução sem evidência.

## Próximo passo

Executar ou revisar a etapa 00: confirmar versões, CUDA/VRAM, tokenizer, template, input_ids, attention_mask, logits, tokens gerados e custo da primeira inferência do Qwen 3B. Registrar no README antes de avançar.