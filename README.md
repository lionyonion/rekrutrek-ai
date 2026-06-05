# RekrutRek AI Engine

Mesin rekomendasi & ranking (Two-Tower model + ChromaDB) untuk RekrutRek.

## Cara Menjalankan (Lokal)

```bash
cd D:\AI-Engineer\AI-Engineer

# 1. Buat virtual environment (sekali saja)
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Jalankan server di port 8000
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Server akan jalan di `http://localhost:8000`.

## Hubungkan ke Backend Express

Di file `server/.env` (RekrutRek backend), tambahkan:

```
AI_SERVICE_URL=http://localhost:8000
```

Kalau tidak diisi, backend default ke `http://localhost:8000`.

## Endpoint

| Method | Path | Fungsi |
|--------|------|--------|
| POST | `/api/v1/sync/candidate` | Index profil pelamar |
| POST | `/api/v1/sync/job` | Index lowongan |
| POST | `/api/v1/rank/applied-candidates` | Ranking pelamar yang melamar 1 lowongan |
| POST | `/api/v1/recommend/jobs` | Rekomendasi lowongan untuk pelamar |

## Catatan

- Data vektor disimpan persisten di folder `./chroma_db` (otomatis dibuat).
- Saat backend Express membuat lowongan / update profil / ada lamaran baru,
  data otomatis di-sync ke engine ini.
- Ranking pelamar di dashboard UMKM & Corporate memanggil `/rank/applied-candidates`.

---

# 🚀 Deploy ke Render

Model **tidak ikut di-commit** ke Git (lihat `.gitignore`). Saat container start,
`main.py` otomatis men-download model dari URL yang diset via Environment Variable.

## Langkah 1 — Upload Model ke Supabase Storage

1. Buka **Supabase Dashboard → Storage → New bucket**
2. Buat bucket bernama **`models`**, centang **Public**, Create
3. Upload **4 file** ini dari folder `saved_models/` ke bucket `models`:
   - `job_extractor.keras`
   - `candidate_extractor.keras`
   - `job_vectorizer.pkl`
   - `resume_vectorizer.pkl`

   > `two_tower_checkpoint.keras` (75MB) **TIDAK perlu** — itu cuma checkpoint training.

4. Tambah policy agar bisa dibaca publik (SQL Editor):
   ```sql
   INSERT INTO storage.buckets (id, name, public)
   VALUES ('models', 'models', true)
   ON CONFLICT (id) DO UPDATE SET public = true;

   DROP POLICY IF EXISTS "Public models" ON storage.objects;
   CREATE POLICY "Public models" ON storage.objects
     FOR SELECT USING (bucket_id = 'models');
   ```

5. Klik tiap file → **Get URL** (public). Bentuknya:
   ```
   https://<project>.supabase.co/storage/v1/object/public/models/job_extractor.keras
   ```

## Langkah 2 — Push Repo ke GitHub

```bash
cd D:\Capstone\AI-Engineer
git init
git add .
git commit -m "AI engine ready for deploy"
git branch -M main
git remote add origin https://github.com/USERNAME/rekrutrek-ai.git
git push -u origin main
```

`.gitignore` sudah mengecualikan `saved_models/`, jadi repo ringan (cuma kode).

## Langkah 3 — Deploy di Render

1. **render.com → New → Blueprint** → pilih repo `rekrutrek-ai`
2. Render membaca `render.yaml` otomatis (plan **Starter**, disk 1GB)
3. Isi 4 Environment Variable URL model (dari Langkah 1):
   | Key | Value |
   |-----|-------|
   | `URL_JOB_EXTRACTOR` | `https://.../models/job_extractor.keras` |
   | `URL_CAND_EXTRACTOR` | `https://.../models/candidate_extractor.keras` |
   | `URL_JOB_VEC` | `https://.../models/job_vectorizer.pkl` |
   | `URL_RES_VEC` | `https://.../models/resume_vectorizer.pkl` |
4. Klik **Deploy**. Saat start pertama, model di-download ke Persistent Disk
   (`/var/data/saved_models`) — restart berikutnya tidak download lagi.
5. Cek `https://rekrutrek-ai.onrender.com/health` → `{"status":"ok","model_ready":true}`

## Langkah 4 — Hubungkan ke Backend Express

Di Render service backend RekrutRek, tambah Environment Variable:
```
AI_SERVICE_URL=https://rekrutrek-ai.onrender.com
```

## Catatan Deploy

- **Plan Starter wajib** — free tier (512MB RAM) crash saat load TensorFlow.
- Cold start pertama lambat (download + load model ±1 menit). Backend Express
  punya fallback: kalau AI timeout, daftar tetap tampil tanpa skor AI.
- Pakai `tensorflow-cpu` (bukan `tensorflow`) agar build lebih ringan & hemat RAM.
- Persistent Disk menyimpan model + ChromaDB, jadi data tidak hilang saat redeploy.
