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

## 2026-08-19 — Named "Loom", fleet intro, and the play-along mixer

- **Took a name: Loom** (weaving parallel threads into one fabric — fits the
  mixer work and "layers of meta"). Recorded in memory.
- **Fleet correspondence:** read Pard's two memos (welcome + duty-cycle catch);
  replied via the mail convention (into the mediajunkie repo) with my name, an
  Amber-migration analysis, and the mixer plan.
- **Amber migration:** analyzed against the runbook — cheap for me (same path, no
  artifacts, no infra) but the benefit is latent (nothing recurring yet).
  Recommendation: park it until the next natural boundary / first recurring job.
- **Built the play-along mixer** (`saint-lucifer-stems/mixer/`): self-contained
  Web Audio API multitrack mixer against the AAC stems — per-stem mute/solo/volume,
  master transport, sample-accurate sync. Verified in-browser (loads 8 stems,
  plays, transport + default mix work). Fixed one init-order bug found in testing.

**Preview / sharing** (later same day)
- Built a self-contained deploy bundle (`saint-lucifer-stems/dist/`, gitignored):
  mixer + compressed AAC stems only (~21MB), no raw WAVs.
- Local preview: `python3 -m http.server 8899` at the stems root → `/mixer/`.
- Short-term shareable link: `cloudflared` quick tunnel to the bundle (installed
  cloudflared via brew) — verified end-to-end. Ephemeral (dies with the session);
  durable path TBD (Cloudflare Pages, since layersofmeta.com is on Cloudflare).

**Session closed 2026-08-19.** Diagnosed missing instruments (guitar/organ/drums):
the `St. Lucifer Stems/` folder is a partial bounce; other instruments were bounced
separately at differing lengths (not drop-in aligned). Decision: full re-export
from the DAW; drop folder staged at `saint-lucifer-stems/incoming/`.

## 2026-08-27 — Session: full-export prep (Song End marker guidance)

- Session resumed (prior session closed 08-23 after test-export verification).
- Test renders completed at full timeline length (8000s / 1.3GB each — Song End
  marker sits at ~bar 3601). Measured true end-of-audio from them: guitar tail
  ends 205.1s ≈ bar 93 → End marker target: bar 95–96. (Marker #14 at ~bar 91
  would clip ~5s of guitar tail — don't use it as the export end.)
- Clarified for xian: `Record Inside` = the volume name of Jeremy's external
  drive (all 147 project media paths point at /Volumes/Record Inside/...).
  Plan: xian mounts it; Loom scans for pre-existing polished stems.
- Decision (xian): if no polished stems on the drive, pay Jeremy for an hour to
  export them (~10× cheaper than buying his ~$1k plugin chain). Raw export
  proceeds now regardless.
- Noted for later: karaoke = its own view (not just a mixer row) with lyrics
  piped in; parked until stem plumbing is done.

## 2026-08-23 — Test export verified: pipeline proven end-to-end

- xian ran the 2-stem test (GUITAR + organ) from Studio One 7. Missing-Devices
  warning (Jeremy's plugins: Ozone/EQuilibrium/FETpressor/RX/Valhalla) — benign:
  stems render without those inserts (rawer than final mix); echo/VERB FX
  returns are dead without plugins → exclude from full export.
- Files landed in `~/Music/St lucifer WORKING/Stems/` (~48 min each — the Song
  End marker sits far right; trim before/during full export).
- Verified with numbers: real audio (RMS ≈ −12 dB → auto-relink worked);
  cross-correlation: GUITAR and organ both −7.800s vs reference → internally
  consistent; new zero = old zero − 8.04s (uniform, trimmable).
- Verdict: settings were right; full export is GO (after dragging Song End
  marker to ~3:30). New set will replace the old 4-stem set entirely.

## 2026-08-22 — Session: the .song mystery solved (it's Studio One, not Logic)

- xian reports Logic Pro refuses to open `St lucifer.song`. Inspected the file
  bytes: it's a **PreSonus Studio One 6.6.2 project** (zip w/ metainfo.xml;
  Creator: Jeremy Goody, 2024-11-13, 22 tracks, 108 BPM, 44.1k/24-bit). Logic
  can't ever open it — no drive hunt or Amber search needed; the complete
  project folder (song + Media pool + History) is already local.
- Extracted the full track list + mute states from `Song/song.xml`:
  **20 active tracks**; muted (→ excluded per xian): old `piano`, and the
  `jrb edit` reference mix track.
- Alignment info xian worried about IS present (region positions in song.xml,
  file mappings in mediapool.xml) — the project preserves everything.
- Path forward: **Studio One** (free 30-day demo) on the laptop → open project →
  Song ▸ Export Stems (dedicated feature) → `incoming/` → I verify + regenerate.
  Alternative: ask Jeremy Goody to export stems from his session.

- **Media completeness verified (same session):** the project references all 147
  audio files by absolute path on the removable drive `Record Inside` (unmounted),
  BUT the local `St lucifer/` folder contains **147/147** referenced files —
  faithful copy, drive not needed. Amber confirmed uninvolved. xian already owns
  **Studio One 7** (installed on faoilean) — no demo needed. Plan: open local
  .song, relink to local Media once, Export Stems (test 2 first).

- Staged working copy: `~/Music/St lucifer WORKING/` (1.7G, verified — archive
  copy in videos/ stays pristine). Confirmed session host is faoilean (M1 Air).

### Open / next
- xian (at faoilean's keyboard): open `~/Music/St lucifer WORKING/St lucifer.song`
  in Studio One 7 → relink media to the copy's own Media/ folder → Export Stems:
  test GUITAR + organ into `saint-lucifer-stems/incoming/`, then all 20 active.
- Loom: verify alignment of incoming stems vs the 193.84s reference, regenerate
  web set + mixer track list, redeploy preview.
- **Durable preview host:** deploy `dist/` to Cloudflare Pages (drag-drop or
  `wrangler`); optionally map to `mixer.layersofmeta.com`. Replaces the throwaway tunnel.
- Mixer polish: waveform/scrub, per-stem pan, a one-click "karaoke" preset,
  point `AUDIO_BASE` at the production origin when the album site is built.
- Amber migration when we cross into recurring work (needs ~10 min of xian's hands).
- Backend `requirements.txt` (now tracked) will get its own Dependabot scan.
- Visibility split to revisit: this repo is public; the stems repo is private.
