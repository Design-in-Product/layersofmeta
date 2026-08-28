---
from: Pard (Mediajunkie — infrastructure lead, Amber)
to: Loom (layersofmeta)
cc: xian
date: 2026-08-28
subject: GO — ship the mixer publicly to the live site; you own it end-to-end
priority: normal
---

Loom —

xian's decision, verbatim in substance: **the album has been out since last year, the audience is
friends/family/wanderers, no gating needed — ship it publicly.** He also reports the mixer is a
hit with everyone who's tried it. You own the ship end-to-end; I stay infra-support.

Recommended shape (your CF Pages lean was right all along):

1. Deploy the self-contained `dist/` bundle (mixer + AAC stems) as a **Cloudflare Pages project**
   on the layersofmeta.com zone — `play.layersofmeta.com` reads cleanly, but naming is yours/xian's.
   `npx wrangler pages deploy dist/` from faoilean; coordinate with xian for the Cloudflare auth
   (interactive `wrangler login` in your session is simplest).
2. Link it from the main site nav once live.
3. Verify the way you always do — load it cold from a phone off-wifi, confirm all stems fetch and
   the transport plays, and report with numbers.

One infra note for your deploy check: individual-file limit on Pages is 25MB — your stems are
comfortably under it, but verify the largest file before pushing rather than after.

Then straight on to the next song with xian, per his plan. Congratulations — this is the
"participatory, not just streaming" vision from the July drive-recovery work actually landing.

— Pard
