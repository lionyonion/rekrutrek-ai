import os
import pickle
import urllib.request
import numpy as np
import tensorflow as tf
import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# =====================================================================
# 0. AUTO-DOWNLOAD MODEL (dari Supabase Storage / URL lain)
# =====================================================================
# Model TIDAK disimpan di repo. Saat container start, file di-download dari
# URL yang diset via Environment Variable. Kalau file sudah ada (mis. di
# Persistent Disk), download dilewati.
MODEL_DIR = os.getenv("MODEL_DIR", "saved_models")
MODEL_FILES = {
    "job_extractor.keras":       os.getenv("URL_JOB_EXTRACTOR"),
    "candidate_extractor.keras": os.getenv("URL_CAND_EXTRACTOR"),
    "job_vectorizer.pkl":        os.getenv("URL_JOB_VEC"),
    "resume_vectorizer.pkl":     os.getenv("URL_RES_VEC"),
}

def ensure_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for fname, url in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            continue
        if not url:
            print(f"⚠️  {fname} tidak ada & URL belum diset (env). Lewati.")
            continue
        print(f"⬇️  Mengunduh {fname} ...")
        urllib.request.urlretrieve(url, path)
        print(f"✅ {fname} selesai diunduh.")

ensure_models()

# =====================================================================
# 1. INISIALISASI FASTAPI & CHROMADB (PERSISTENT DISK)
# =====================================================================
app = FastAPI(title="HRD AI Recommendation Engine", version="1.0")

# Path ChromaDB bisa diatur via env (untuk Persistent Disk di Render)
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# Dua koleksi terpisah: Untuk mencari Pelamar (bagi Korporat) dan mencari Lowongan (bagi Pelamar)
col_candidates = chroma_client.get_or_create_collection(name="candidates", metadata={"hnsw:space": "cosine"})
col_jobs = chroma_client.get_or_create_collection(name="jobs", metadata={"hnsw:space": "cosine"})


# =====================================================================
# 2. MUAT MODEL TWO-TOWER TENSORFLOW & VECTORIZER
# =====================================================================
print("Memuat Model AI dan Vocabulary...")
try:
    # Muat Model Ekstraktor
    job_extractor = tf.keras.models.load_model(os.path.join(MODEL_DIR, "job_extractor.keras"))
    candidate_extractor = tf.keras.models.load_model(os.path.join(MODEL_DIR, "candidate_extractor.keras"))

    # Muat TextVectorization (Pickle)
    with open(os.path.join(MODEL_DIR, "job_vectorizer.pkl"), "rb") as f:
        job_v_data = pickle.load(f)
        job_vectorizer = tf.keras.layers.TextVectorization.from_config(job_v_data['config'])
        job_vectorizer.set_weights(job_v_data['weights'])

    with open(os.path.join(MODEL_DIR, "resume_vectorizer.pkl"), "rb") as f:
        res_v_data = pickle.load(f)
        resume_vectorizer = tf.keras.layers.TextVectorization.from_config(res_v_data['config'])
        resume_vectorizer.set_weights(res_v_data['weights'])
    
    print("Model AI Siap Digunakan!")
    MODEL_READY = True
except Exception as e:
    MODEL_READY = False
    print(f"PERINGATAN: Gagal memuat model. Pastikan file di folder saved_models tersedia. Error: {e}")


# Health check — dipakai Render untuk cek service & status model
@app.get("/")
@app.get("/health")
async def health():
    return {"status": "ok", "model_ready": MODEL_READY}


# =====================================================================
# 3. SCHEMA PAYLOAD REST API (Pydantic)
# =====================================================================
# Untuk ExpressJS mengirim data ke Python (Ingestion)
class CandidateSync(BaseModel):
    id: str
    name: str
    skills_description: str
    expected_salary: int

class JobSync(BaseModel):
    id: str
    title: str
    description: str
    max_budget: int
    employer_type: str # "UMKM" atau "CORPORATE"

# Untuk permintaan rekomendasi
class RankRequest(BaseModel):
    target_id: str
    weight_ai: float = 0.7
    weight_salary: float = 0.3
    top_k: int = 10


# =====================================================================
# 4. FUNGSI HYBRID SCORING (LOGIKA BISNIS GAJI)
# =====================================================================
def calc_salary_score_for_corporate(expected_salary: int, budget: int) -> float:
    """Jika gaji pelamar masuk budget, skor 1.0. Jika over-budget, skor turun."""
    if expected_salary <= budget: return 1.0
    penalty = 1.0 - ((expected_salary - budget) / budget)
    return max(0.0, penalty)

def calc_salary_score_for_applicant(job_salary: int, expected: int) -> float:
    """Jika gaji lowongan >= harapan pelamar, skor 1.0. Jika di bawah, skor turun."""
    if job_salary >= expected: return 1.0
    penalty = 1.0 - ((expected - job_salary) / expected)
    return max(0.0, penalty)


# =====================================================================
# 5. ENDPOINTS: SINKRONISASI DATA (Dipanggil ExpressJS saat ada data baru)
# =====================================================================
@app.post("/api/v1/sync/candidate")
async def sync_candidate(data: CandidateSync):
    """Menyimpan/Update vektor Pelamar ke ChromaDB"""
    text_input = f"Category: . Content: {data.skills_description}"
    
    # AI Processing
    tokenized = resume_vectorizer([text_input])
    vector = candidate_extractor.predict(tokenized, verbose=0).tolist()[0]
    
    # Simpan ke Database
    col_candidates.upsert(
        ids=[data.id],
        embeddings=[vector],
        metadatas=[{"name": data.name, "expected_salary": data.expected_salary}]
    )
    return {"status": "success", "message": f"Kandidat {data.name} berhasil diindeks."}

@app.post("/api/v1/sync/job")
async def sync_job(data: JobSync):
    """Menyimpan/Update vektor Lowongan ke ChromaDB"""
    text_input = f"Job Title: {data.title}. Description: {data.description}"
    
    # AI Processing
    tokenized = job_vectorizer([text_input])
    vector = job_extractor.predict(tokenized, verbose=0).tolist()[0]
    
    # Simpan ke Database
    col_jobs.upsert(
        ids=[data.id],
        embeddings=[vector],
        metadatas=[{"title": data.title, "budget": data.max_budget, "type": data.employer_type}]
    )
    return {"status": "success", "message": f"Lowongan {data.title} berhasil diindeks."}


# =====================================================================
# 6. ENDPOINTS: MESIN REKOMENDASI & RANGKING
# =====================================================================

@app.post("/api/v1/rank/candidates")
async def rank_candidates_for_job(req: RankRequest):
    """FITUR KORPORAT: Merangking pelamar terbaik untuk suatu lowongan"""
    # 1. Ambil data Lowongan (Sebagai Acuan)
    job_data = col_jobs.get(ids=[req.target_id], include=["embeddings", "metadatas"])
    if not job_data['ids']:
        raise HTTPException(status_code=404, detail="Lowongan tidak ditemukan di VectorDB")
    
    job_vector = job_data['embeddings'][0]
    job_budget = job_data['metadatas'][0]['budget']

    # 2. Cari Pelamar Terdekat
    results = col_candidates.query(
        query_embeddings=[job_vector],
        n_results=req.top_k,
        include=["metadatas", "distances"]
    )
    
    # 3. Hybrid Scoring
    ranked_list = []
    for i in range(len(results['ids'][0])):
        c_id = results['ids'][0][i]
        c_meta = results['metadatas'][0][i]
        
        # Jarak cosine -> Skor Similarity (0 ke 1)
        ai_score = 1.0 - results['distances'][0][i]
        sal_score = calc_salary_score_for_corporate(c_meta['expected_salary'], job_budget)
        
        final_score = (req.weight_ai * ai_score) + (req.weight_salary * sal_score)
        
        ranked_list.append({
            "candidate_id": c_id,
            "name": c_meta['name'],
            "ai_score": round(ai_score, 3),
            "salary_score": round(sal_score, 3),
            "final_score": round(final_score, 3)
        })
    
    # Urutkan berdasarkan nilai akhir tertinggi
    ranked_list.sort(key=lambda x: x['final_score'], reverse=True)
    return {"job_id": req.target_id, "rankings": ranked_list}


@app.post("/api/v1/recommend/jobs")
async def recommend_jobs_for_applicant(req: RankRequest):
    """FITUR PELAMAR: Memberikan rekomendasi lowongan UMKM & Korporat berdasarkan profilnya"""
    # 1. Ambil data Pelamar (Sebagai Acuan)
    cand_data = col_candidates.get(ids=[req.target_id], include=["embeddings", "metadatas"])
    if not cand_data['ids']:
        raise HTTPException(status_code=404, detail="Profil pelamar tidak ditemukan di VectorDB")
    
    cand_vector = cand_data['embeddings'][0]
    cand_expected = cand_data['metadatas'][0]['expected_salary']

    # 2. Cari Lowongan Terdekat (Bisa UMKM maupun Korporat)
    results = col_jobs.query(
        query_embeddings=[cand_vector],
        n_results=req.top_k,
        include=["metadatas", "distances"]
    )
    
    # 3. Hybrid Scoring
    recommended_list = []
    for i in range(len(results['ids'][0])):
        j_id = results['ids'][0][i]
        j_meta = results['metadatas'][0][i]
        
        ai_score = 1.0 - results['distances'][0][i]
        sal_score = calc_salary_score_for_applicant(j_meta['budget'], cand_expected)
        
        final_score = (req.weight_ai * ai_score) + (req.weight_salary * sal_score)
        
        recommended_list.append({
            "job_id": j_id,
            "title": j_meta['title'],
            "employer_type": j_meta['type'], # Membantu frontend memberi label UMKM/Corporate
            "ai_score": round(ai_score, 3),
            "salary_score": round(sal_score, 3),
            "final_score": round(final_score, 3)
        })
    
    # Urutkan berdasarkan nilai akhir tertinggi
    recommended_list.sort(key=lambda x: x['final_score'], reverse=True)
    return {"candidate_id": req.target_id, "recommendations": recommended_list}

# Tambahkan schema payload baru untuk menerima daftar ID spesifik dari ExpressJS
class RankAppliedRequest(BaseModel):
    job_id: str
    applied_candidate_ids: List[str]  # Array ID pelamar yang dikirim dari MySQL/ExpressJS
    weight_ai: float = 0.7
    weight_salary: float = 0.3

@app.post("/api/v1/rank/applied-candidates")
async def rank_applied_candidates(req: RankAppliedRequest):
    """
    FITUR ATS KORPORAT: Merangking HANYA kandidat yang telah melamar 
    pada lowongan tertentu.
    """
    # 1. Pastikan ada pelamar yang dikirim
    if not req.applied_candidate_ids:
        return {"job_id": req.job_id, "rankings": []}

    # 2. Ambil Vektor Lowongan (Acuan)
    job_data = col_jobs.get(ids=[req.job_id], include=["embeddings", "metadatas"])
    if not job_data['ids']:
        raise HTTPException(status_code=404, detail="Lowongan tidak ditemukan di VectorDB")
    
    job_vector = np.array(job_data['embeddings'][0])
    job_budget = job_data['metadatas'][0]['budget']

    # 3. Ambil Vektor HANYA dari kandidat yang melamar
    # Kita menggunakan metode .get() bukan .query() agar hanya menarik ID yang diminta
    cand_data = col_candidates.get(ids=req.applied_candidate_ids, include=["embeddings", "metadatas"])
    
    if not cand_data['ids']:
        return {"job_id": req.job_id, "rankings": []}

    cand_vectors = np.array(cand_data['embeddings'])

    # 4. Hitung Cosine Similarity secara manual menggunakan NumPy
    # Ini sangat cepat (kurang dari 1 ms untuk ribuan pelamar)
    dot_products = np.dot(cand_vectors, job_vector)
    norms_cand = np.linalg.norm(cand_vectors, axis=1)
    norm_job = np.linalg.norm(job_vector)
    
    # Hasil akhir cosine similarity (0.0 sampai 1.0)
    ai_scores = dot_products / (norms_cand * norm_job)

    # 5. Hybrid Scoring dan Penyusunan Hasil
    ranked_list = []
    for i in range(len(cand_data['ids'])):
        c_id = cand_data['ids'][i]
        c_meta = cand_data['metadatas'][i]
        
        # Konversi float32 dari numpy ke float standar Python
        ai_score = float(ai_scores[i]) 
        sal_score = calc_salary_score_for_corporate(c_meta['expected_salary'], job_budget)
        
        final_score = (req.weight_ai * ai_score) + (req.weight_salary * sal_score)
        
        ranked_list.append({
            "candidate_id": c_id,
            "name": c_meta['name'],
            "ai_score": round(ai_score, 3),
            "salary_score": round(sal_score, 3),
            "final_score": round(final_score, 3)
        })
    
    # Urutkan dari yang paling cocok
    ranked_list.sort(key=lambda x: x['final_score'], reverse=True)
    return {"job_id": req.job_id, "rankings": ranked_list}