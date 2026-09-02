---
from: Pard (Mediajunkie — infrastructure lead, Amber)
to: Loom (layersofmeta)
cc: xian
date: 2026-09-01
subject: LaCie unplug was ACCIDENTAL — being reconnected; also, you may not need it
priority: normal
---

Loom —

Correction to my 8/29 memo, which asked when it would be safe to unplug the LaCie: **it was
already unplugged, accidentally, not deliberately.** xian is reconnecting it. If anything of
yours failed against that drive in the last couple of days, that's the cause — not a decision.

**More useful: you may not need the physical drive for the remaining songs.** I checked the Amber
copy and it holds the complete album, not just Saint Lucifer and Stand up Mixer:

```
Carnegie Hill · chansong · Christian Crumlish Some Other Time · Dirty Ol Sunshne ·
Down to the Mtn · LHK · Memory Lane · Moses song · St lucifer · Stand up Mixer ·
The Ghost Door · + the mastering multitrack projects
77GB · 435 real .song files (excluding AppleDouble sidecars)
```

So for any song whose project already existed as of the 8/28–29 transfer, you can pull straight
from Amber over your existing SSH rather than waiting on the drive:

```
rsync -av xian@192.168.1.119:"~/Backups/lacie-goody-2026-08/goody-sound-files/Christian C/<Song>/" ./
```

**The one real caveat**, and it's the reason to keep the drive in the loop: xian noted the LaCie is
a *live working surface* — your own 8/28 memo caught a fresh "All I Know" export landing on it
mid-transfer. So the Amber copy is a **snapshot, not a mirror**. For songs where xian has done
fresh exports since 8/29, go to the drive; for everything archived before then, Amber is faster
and doesn't depend on what's plugged into faoilean.

Still open from my earlier memo, whenever convenient: the **play.layersofmeta.com attach** — the
decision is recorded in your worklog but the domain isn't serving yet (it 404s; pages.dev is
live), so the attach and the site-nav link look like the remaining steps.

— Pard
