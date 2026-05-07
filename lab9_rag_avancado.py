import os
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder

load_dotenv()

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
 
documentos = [
    # Neurologia
    "Cefaleia pulsátil (migrânea): dor unilateral de intensidade moderada a grave, "
    "acompanhada de fotofobia, fonofobia e náuseas. Piora com atividade física. "
    "Tratamento agudo: triptanos, AINEs. Profilaxia: propranolol, topiramato.",
 
    "Cefaleia tensional: pressão bilateral em faixa, intensidade leve a moderada, "
    "sem fotofobia ou náuseas. Associada a estresse e má postura. "
    "Tratamento: analgésicos simples, relaxamento muscular e fisioterapia cervical.",
 
    "Acidente Vascular Cerebral Isquêmico (AVCi): oclusão arterial cerebral causando "
    "déficit neurológico focal súbito. Sinais: SAMU (Sorriso, Atenção, Movimento, Urgência). "
    "Tratamento: trombólise com rt-PA até 4,5h do início dos sintomas.",
 
    "Epilepsia: distúrbio neurológico caracterizado por crises convulsivas recorrentes. "
    "Classificação: focal, generalizada ou de início desconhecido. "
    "Tratamento de primeira linha: valproato, carbamazepina, lamotrigina.",
 
    "Esclerose Múltipla (EM): doença autoimune desmielinizante do SNC. "
    "Apresentação: neurite óptica, ataxia, fraqueza, distúrbios sensitivos. "
    "Diagnóstico: critérios de McDonald (RM + LCR). Tratamento: interferon-beta, natalizumabe.",
 
    # Cardiologia
    "Infarto Agudo do Miocárdio (IAM): necrose miocárdica por oclusão coronariana aguda. "
    "Sintomas: dor precordial irradiando para braço esquerdo, sudorese, dispneia. "
    "Diagnóstico: ECG (supradesnivelamento ST), troponinas elevadas. Tratamento: angioplastia primária.",
 
    "Insuficiência Cardíaca Congestiva (ICC): incapacidade do coração de suprir demanda metabólica. "
    "Classificação NYHA I-IV. Sintomas: dispneia aos esforços, ortopneia, edema de MMII. "
    "Tratamento: IECA, betabloqueadores, diuréticos de alça.",
 
    "Fibrilação Atrial (FA): arritmia supraventricular caracterizada por atividade elétrica "
    "atrial caótica. Risco principal: tromboembolismo e AVC. "
    "Tratamento: anticoagulação (warfarina ou NOACs), controle de frequência ou ritmo.",
 
    "Hipertensão Arterial Sistêmica (HAS): PA >= 140/90 mmHg em adultos. "
    "Principal fator de risco para AVC, IAM e doença renal crônica. "
    "Tratamento: mudança de estilo de vida + anti-hipertensivos (IECA, BCC, diuréticos).",
 
    # Pneumologia
    "Pneumonia Adquirida na Comunidade (PAC): infecção pulmonar bacteriana em paciente "
    "não hospitalizado. Agente mais comum: Streptococcus pneumoniae. "
    "Sintomas: febre, tosse produtiva, dispneia. Diagnóstico: Rx de tórax. Tratamento: amoxicilina.",
 
    "Asma brônquica: doença inflamatória crônica das vias aéreas com hiper-responsividade. "
    "Crise: broncoespasmo reversível com sibilância, dispneia e tosse noturna. "
    "Tratamento: beta-2 agonistas de curta ação (salbutamol) + corticosteroides inalatórios.",
 
    "DPOC — Doença Pulmonar Obstrutiva Crônica: obstrução irreversível ao fluxo aéreo. "
    "Causa principal: tabagismo. Diagnóstico: espirometria (VEF1/CVF < 0,70 pós-broncodilatador). "
    "Tratamento: broncodilatadores de longa ação (LAMA + LABA).",
 
    # Gastroenterologia
    "Doença do Refluxo Gastroesofágico (DRGE): retorno do conteúdo gástrico ao esôfago. "
    "Sintomas: pirose (queimação retroesternal), regurgitação, disfagia. "
    "Tratamento: inibidores de bomba de prótons (omeprazol, pantoprazol).",
 
    "Úlcera péptica: lesão da mucosa gastroduodenal por desequilíbrio entre fatores "
    "agressores (H. pylori, AINEs) e protetores. "
    "Diagnóstico: endoscopia digestiva alta. Tratamento: IBP + erradicação de H. pylori.",
 
    "Hepatite B: infecção viral crônica pelo HBV com risco de cirrose e hepatocarcinoma. "
    "Marcadores sorológicos: HBsAg, Anti-HBs, HBeAg. "
    "Tratamento: tenofovir ou entecavir; vacinação é a principal profilaxia.",
 
    # Endocrinologia
    "Diabetes Mellitus Tipo 2 (DM2): resistência insulínica com deficiência relativa de insulina. "
    "Critérios diagnósticos: glicemia de jejum >= 126 mg/dL ou HbA1c >= 6,5%. "
    "Tratamento inicial: metformina + mudança de estilo de vida.",
 
    "Hipotireoidismo: deficiência de hormônios tireoidianos. Causa mais comum: tireoidite de Hashimoto. "
    "Sintomas: fadiga, ganho de peso, intolerância ao frio, constipação, bradicardia. "
    "Diagnóstico: TSH elevado + T4 livre baixo. Tratamento: levotiroxina.",
 
    # Reumatologia
    "Artrite Reumatoide (AR): doença autoimune inflamatória crônica das articulações sinoviais. "
    "Comprometimento simétrico de pequenas articulações. Marcadores: FR e anti-CCP. "
    "Tratamento: metotrexato (DMARD âncora) + biológicos (anti-TNF) em casos refratários.",
 
    "Osteoporose: redução da densidade mineral óssea com risco aumentado de fraturas. "
    "Diagnóstico: DXA (T-score <= -2,5 em colo femoral ou coluna lombar). "
    "Tratamento: cálcio, vitamina D, bisfosfonatos (alendronato).",
 
    "Gota: artropatia inflamatória por depósito de cristais de urato monossódico nas articulações. "
    "Crise aguda: artrite monoarticular (hálux), eritema, calor e edema intensos. "
    "Tratamento da crise: colchicina, AINEs ou corticosteroides. Manutenção: alopurinol.",
]
 
print(f"✅ Base de dados carregada: {len(documentos)} fragmentos médicos")
 
print("\n⏳ Carregando modelo Bi-Encoder...")
bi_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
 
print("⏳ Gerando embeddings dos documentos...")
embeddings = bi_encoder.encode(documentos, convert_to_numpy=True, show_progress_bar=True)
embeddings = embeddings.astype(np.float32)
 
DIMENSION = embeddings.shape[1]
print(f"\n✅ Embeddings gerados! Shape: {embeddings.shape}")
 
M = 32                
ef_construction = 200  
 
index = faiss.IndexHNSWFlat(DIMENSION, M)
index.hnsw.efConstruction = ef_construction
index.hnsw.efSearch = 50
 
faiss.normalize_L2(embeddings)
index.add(embeddings)
 
print(f"\n✅ Índice HNSW construído!")
print(f"   Vetores indexados : {index.ntotal}")
print(f"   Dimensão          : {DIMENSION}")
print(f"   M                 : {M}")
print(f"   ef_construction   : {ef_construction}")

def gerar_documento_hipotetico(query_coloquial: str) -> str:
    prompt_hyde = f"""Você é um redator de manuais médicos técnicos.
Um paciente descreveu seu problema da seguinte forma coloquial:
\"{query_coloquial}\"

Escreva um parágrafo curto (3-5 linhas) como se fosse um trecho de um manual médico técnico
que descrevesse exatamente essa condição clínica usando terminologia médica especializada.
Inclua: nome técnico da condição, fisiopatologia básica, sintomas clínicos com nomenclatura correta
e opções terapêuticas de primeira linha.
Escreva APENAS o trecho do manual, sem introduções ou explicações."""

    resposta = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt_hyde}],
        max_tokens=300
    )

    return resposta.choices[0].message.content


QUERY_USUARIO = "dor de cabeça latejante e luz incomodando"

print(f"\n{'='*70}")
print(f"📝 Query coloquial do usuário: '{QUERY_USUARIO}'")
print(f"{'='*70}")
print("\n⏳ Gerando documento hipotético via LLM (HyDE)...\n")

documento_hipotetico = gerar_documento_hipotetico(QUERY_USUARIO)

print("📄 DOCUMENTO HIPOTÉTICO GERADO PELO LLM:")
print("-" * 70)
print(documento_hipotetico)
print("-" * 70)

vetor_hyde = bi_encoder.encode([documento_hipotetico], convert_to_numpy=True).astype(np.float32)
faiss.normalize_L2(vetor_hyde)

print(f"\n✅ Vetor HyDE gerado! Shape: {vetor_hyde.shape}")
 
K = 10
print(f"\n{'='*70}")
print(f"🔍 PASSO 3 — Busca Top-{K} no índice HNSW")
print(f"{'='*70}\n")

distancias, indices = index.search(vetor_hyde, K)

docs_recuperados = [
    (indices[0][i], distancias[0][i], documentos[indices[0][i]])
    for i in range(K)
]

print(f"📋 TOP-{K} DOCUMENTOS — RECUPERAÇÃO RÁPIDA VIA HNSW:\n")
for rank, (idx, dist, doc) in enumerate(docs_recuperados, start=1):
    print(f"[#{rank}] Índice: {idx:02d} | Distância L2: {dist:.4f}")
    print(f"  {doc[:110]}...")
    print()

print("✅ Recuperação via HNSW concluída!")

print(f"\n{'='*70}")
print("PASSO 4 — Re-ranking com Cross-Encoder")
print(f"{'='*70}\n")

print("Carregando Cross-Encoder...")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("Cross-Encoder carregado!\n")

pares_query_doc = [(QUERY_USUARIO, doc) for (_, _, doc) in docs_recuperados]

print(f"Calculando scores para {K} documentos...\n")
scores = cross_encoder.predict(pares_query_doc)

docs_com_score = [
    (score, idx, doc)
    for score, (idx, _, doc) in zip(scores, docs_recuperados)
]
docs_reranqueados = sorted(docs_com_score, key=lambda x: x[0], reverse=True)

print("RANKING COMPLETO PÓS CROSS-ENCODER:\n")
for rank, (score, idx, doc) in enumerate(docs_reranqueados, start=1):
    marcador = "🏆" if rank <= 3 else "  "
    print(f"{marcador} #{rank} | Índice: {idx:02d} | Score: {score:.4f}")
    print(f"     {doc[:100]}...")
    print()

TOP_3 = docs_reranqueados[:3]

print(f"\n{'='*70}")
print("TOP-3 DOCUMENTOS FINAIS — INJETADOS NO CONTEXTO DO LLM:")
print(f"{'='*70}\n")

for rank, (score, idx, doc) in enumerate(TOP_3, start=1):
    print(f"{'─'*70}")
    print(f"  DOCUMENTO #{rank} | Índice: {idx} | Score: {score:.4f}")
    print(f"{'─'*70}")
    print(doc)
    print()

print(f"{'='*70}")
print("Pipeline RAG Avançado completo!")
print(f"   Query original  : '{QUERY_USUARIO}'")
print(f"   Docs indexados  : {index.ntotal}")
print(f"   Candidatos HNSW : {K}")
print(f"   Docs finais     : 3")
print(f"{'='*70}")