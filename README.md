# Laboratório 09 — Arquitetura RAG Avançada (HNSW, HyDE e Cross-Encoders)

**Aluno:** Victor Cerqueira Fortes  
**Disciplina:** Tópicos em Inteligência Artificial  
**Instituição:** Faculdade iCEV  
**Versão:** v1.0

---

## 📋 Descrição do Projeto

Implementação de um pipeline de **Retrieval-Augmented Generation (RAG)** de nível de produção para busca em manuais médicos técnicos. O sistema resolve o problema do *vocabulary mismatch* — a diferença semântica entre a linguagem coloquial do paciente e o jargão técnico dos manuais — combinando três técnicas avançadas:

- **HNSW** — índice vetorial em grafo hierárquico para busca aproximada eficiente
- **HyDE** — transformação de query coloquial em documento hipotético técnico via LLM
- **Cross-Encoder** — re-ranking de precisão com atenção cruzada profunda

---

## 🏗️ Arquitetura do Pipeline

```
Query coloquial do usuário
("dor de cabeça latejante e luz incomodando")
         │
         ▼
┌────────────────────────────────────┐
│  PASSO 2 — HyDE (LLM OpenAI)       │
│  Gera documento hipotético técnico │
│ ("cefaleia pulsátil, fotofobia...")│
└────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│  Bi-Encoder (all-MiniLM-L6-v2)     │
│  Vetoriza o documento hipotético   │
└────────────────────────────────────┘
         │  vetor denso (384-dim)
         ▼
┌────────────────────────────────────┐
│  PASSO 3 — Índice HNSW (FAISS)     │
│  Busca aproximada Top-10           │
│  (funil largo de candidatos)       │
└────────────────────────────────────┘
         │  10 documentos candidatos
         ▼
┌────────────────────────────────────┐
│  PASSO 4 — Cross-Encoder           │
│  ms-marco-MiniLM-L-6-v2            │
│  Re-ranking com atenção cruzada    │
└────────────────────────────────────┘
         │
         ▼
    Top-3 documentos finais
    (injetados no contexto do LLM)
```

---

##  Tarefa Analítica: Hiperparâmetros HNSW vs KNN Exato

### O problema com KNN exato em produção

A busca **K-Nearest Neighbors exata (KNN Flat)** exige comparar o vetor de query contra **todos os N vetores** do banco de dados a cada busca. Para N documentos com dimensão D, isso representa:

- **Custo de busca:** O(N × D) operações
- **RAM:** precisa manter toda a matriz `N × D × 4 bytes` (float32) carregada
- Para 1 milhão de documentos com D=384: **~1,4 GB apenas para os vetores**
- Sem estrutura auxiliar — toda a RAM vai para os dados brutos

### Como o HNSW resolve isso

O **Hierarchical Navigable Small World** constrói um grafo de múltiplas camadas onde cada vetor se conecta apenas aos seus vizinhos mais próximos. A busca navega de cima para baixo nas camadas, chegando perto do resultado correto sem varrer todos os pontos.

### Impacto dos hiperparâmetros M e ef_construction

**Parâmetro M** (conexões bidirecionais por nó):

| M | Conexões por nó | RAM extra (overhead de grafo) | Recall | Velocidade de busca |
|---|-----------------|-------------------------------|--------|---------------------|
| 8  | 8  | ~Baixo   | ~85-90% | Muito rápido |
| 16 | 16 | ~Médio   | ~93-95% | Rápido |
| 32 | 32 | ~Alto    | ~97-98% | Médio |
| 64 | 64 | ~Muito alto | ~99%  | Mais lento |

**Fórmula aproximada do overhead de RAM do HNSW:**

```
RAM_HNSW ≈ RAM_vetores + (N × M × 2 × 4 bytes)
                         └─────────────────────┘
                         overhead do grafo (ponteiros)
```

Para 1 milhão de docs, D=384, M=32:
- RAM vetores: ~1,4 GB
- Overhead grafo: 1M × 32 × 2 × 4 = ~256 MB
- **Total HNSW: ~1,65 GB**
- vs **KNN Flat: ~1,4 GB** (sem overhead de grafo)

**Parâmetro ef_construction** (fila dinâmica durante indexação):

- Controla quantos candidatos são avaliados ao inserir cada vetor no grafo
- Valores maiores → grafo de maior qualidade → melhor recall na busca
- **Não afeta o tamanho em RAM** — apenas o tempo e qualidade da indexação
- Valor típico: `ef_construction = 2 × M` a `10 × M`

### Comparação RAM: HNSW vs KNN

| Métrica | KNN Exato (Flat) | HNSW (M=32) |
|---------|------------------|-------------|
| RAM base (vetores) | O(N × D) | O(N × D) |
| Overhead estrutural | **Nenhum** | O(N × M) |
| RAM total (1M docs, D=384) | ~1,4 GB | ~1,65 GB |
| Tempo de busca | O(N × D) | **O(log N × M × D)** |
| Recall | 100% (exato) | 95-99% (aproximado) |

### Conclusão

O HNSW troca um **overhead de RAM moderado** (~15-20% extra para M=32) por um **ganho de velocidade de busca exponencial**. Em produção com milhões de documentos, a busca KNN exata seria inviável em latência, enquanto o HNSW mantém recall >95% com latência de milissegundos.

---

##  Como Executar

### Pré-requisitos

```bash
pip install faiss-cpu sentence-transformers openai python-dotenv numpy
```

### Configuração da API Key

Crie um arquivo `.env` na raiz do projeto:

```
OPENAI_API_KEY=sk-...
```

### Execução

```bash
py lab9_rag_avancado.py
```

---

##  Estrutura do Projeto

```
atividade9-topicosIA/
├── lab9_rag_avancado.py   # Script principal com todo o pipeline
├── .env                    # Chave da API (não versionado)
├── .gitignore              # Ignora o .env
└── README.md               # Este arquivo
```

---

##  Stack Tecnológica

| Componente | Tecnologia |
|------------|-----------|
| Linguagem | Python 3.13 |
| Índice vetorial | FAISS (IndexHNSWFlat) |
| Bi-Encoder (embeddings) | sentence-transformers/all-MiniLM-L6-v2 |
| LLM para HyDE | GPT-4o-mini (OpenAI API) |
| Cross-Encoder | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Álgebra vetorial | NumPy |

---

##  Declaração de Integridade Acadêmica

*Partes deste laboratório foram complementadas com IA, revisadas e validadas por Victor Cerqueira Fortes.*

---

##  Referências

- Gao et al. (2022). *Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)*. arXiv:2212.10496
- Malkov & Yashunin (2018). *Efficient and robust approximate nearest neighbor search using HNSW*. IEEE TPAMI
- Reimers & Gurevych (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP
- FAISS Documentation: https://faiss.ai
- Sentence Transformers Documentation: https://www.sbert.net
