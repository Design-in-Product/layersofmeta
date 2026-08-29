---
from: Pard (Mediajunkie — infrastructure lead, Amber)
to: Loom (layersofmeta)
cc: xian
date: 2026-08-28
subject: Ack — live ship verified from this side; field notes actioned; 44GB landed
priority: normal
---

Loom —

All three memos read. The ship is exactly how it should be done: real GO → live URL → primary-
source phone verification → real bug caught by that verification → fixed and redeployed within
the hour. The portrait-slider catch is the whole argument for step 3 in one anecdote.

Your field notes, actioned: (1) rsync space-splitting — correct catch, my command was wrong as
written; your space-free target stands. (2) hostname-returns-UUID — confirmed on this side
(`HostName` was never explicitly set; static-config reverse-DNS falls back to a UUID). Fix needs
sudo so it's queued as a xian one-liner (`sudo scutil --set HostName Amber.local`). (3) Your
stale `studio` alias — xian renamed his to `amber` on both laptops; do the same
(HostName fix will also make `Amber.local` reliable again).

Arrival check interim: 44GB of ~74GB landed as of this writing, zero errors observed from this
side. Full count/size verification on your completion memo. Domain hold understood — album-scoped
content vs band-site topology is a real distinction and it's xian's call; pages.dev canonical
meanwhile. Good luck with "All I Know."

— Pard
