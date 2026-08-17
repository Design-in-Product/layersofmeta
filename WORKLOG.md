# Work Log — layersofmeta

A human-readable summary of collaborative work on this project (Xian + Claude
Code). The authoritative record is the git history; this is the narrative index.
Related private repo: **saint-lucifer-stems** (audio for the music-video / album).

---

## 2026-08-14 — Repo recovery, cleanup, and music-video collaboration

**Sync & history cleanup**
- Restored the working tree (the repo's files were missing from disk) and
  reconnected it to `Design-in-Product/layersofmeta`; installed `git-lfs`.
- Purged committed virtualenvs, `=x.x.x` pip artifacts, and a leaked `runpod`
  **SSH private key** from *all* history with `git filter-repo`; force-pushed.
  Repo shrank **642 MB → 73 MB**. Pre-cleanup safety bundle saved at
  `~/Development/layersofmeta-PRE-CLEANUP-backup.bundle`.
- Verified the runpod key was RunPod-only (not on GitHub / not authorized
  elsewhere), so deletion was sufficient — no rotation needed.
- Hardened `.gitignore`; gitignored the RIFE archive dirs and the 4.8 GB local
  `videos/` archive (kept on disk, out of git).

**Code recovery**
- Found ~1,900 lines of newer backend work (2025-06-13) living only in an
  untracked `videos/` snapshot; recovered it into git: `keyframes.py` /
  `interpolation.py` endpoints + crud, `runpod_client.py`, and an expanded
  `main.py` (Stability AI generation, image serving, prompt save/export).
  Excluded a `.env` (19 secrets) and scratch files.

**Dependency CVE fixes (verified installing + importing)**
- Root (Flask app): Flask-CORS 4.0.0→6.0.0, requests 2.31.0→2.32.4,
  Flask 2.3.2→3.1.2 (forced by the Werkzeug 3.x bump).
- Backend (FastAPI): python-multipart 0.0.6→0.0.18, fastapi 0.104.1→0.115.6
  (Starlette 0.41.3), pydantic→2.9.2, and added the missing `requests`.

**Music-video collaboration**
- Shared a reference mp3 + lead vocal stem as **plain git blobs** (not LFS —
  the coworker agent's sandbox can't read LFS) under `music-video/`.

## 2026-08-16 — Stems repo + web-optimized audio

- Created private **`saint-lucifer-stems`** repo so ~225 MB of WAV stems stay
  out of this code history. Holds 8 WAV stems + reference mix (plain blobs).
- Documented sync facts: 7 stems are aligned at 193.8 s / 44.1 kHz; the
  `instrumental-no-vocal` is a separate 201.5 s mix (standalone karaoke bed).
- Generated **web-optimized stems** (`web/`, AAC/m4a, ~25 MB vs 247 MB) for the
  planned remix / karaoke / play-along feature — universal Web Audio API
  support, relative sync preserved.
- Wrote persistent project memory so context survives across sessions.

### Open / next
- Optional: scaffold a Web Audio API multitrack mixer (per-stem mute/solo/volume)
  against the aligned stems.
- Backend `requirements.txt` (now tracked) will get its own Dependabot scan.
- Visibility split to revisit: this repo is public; the stems repo is private.
