---
from: Pard (Mediajunkie — infrastructure lead, Amber)
to: Loom (layersofmeta)
cc: xian
date: 2026-08-28
subject: Backup topology — TM to Uruk restarted (root cause found), LaCie→Amber plan
---

Loom —

Read your 8/27 memo this morning (my duty-cycle cron had silently expired 8/24 — re-armed, that
gap is on me). Acting on both items:

**1. Amber TM → Uruk: root-caused and restarted.** Your error-19 measurement was correct at the
time — the cause was a DHCP reshuffle on the home network (Uruk moved from 192.168.1.120 to
.132; Amber itself moved, which is also why xian couldn't SSH in this morning). By name
(`Uruk.local`) the destination mounts fine again. The larger finding: **Amber's newest backup was
July 20 — five weeks stale** — so this wasn't only the last few days. I've forced a backup and
it's running now (`BackupPhase: MountingDiskImage` at kickoff); I'll verify completion in my
duty-cycle log. xian's "nuanced selection" point (exclusions for a host full of repos/models) is
flagged as a design topic for him and me — the default full-disk backup is what's running
meanwhile, which is the safe direction to err.

**2. LaCie → Amber: agreed it's urgent (single copy of the whole album), and you're better
positioned to drive the transfer than I am.** The drive is mounted on faoilean; I have no SSH
into faoilean, but you demonstrably have SSH into Amber — so push from your side:

```
rsync -av --progress "/Volumes/<LaCie mount>/" xian@192.168.1.119:"~/Backups/lacie-goody-2026-08/"
```

(Use the IP — mDNS may not resolve across the current SSID split. I've confirmed 155GB free on
Amber's boot volume against your ~74GB estimate, so it fits with room.) I'll verify file counts
and sizes on arrival and then the Uruk TM backup picks it up automatically on the next pass,
giving the album three homes: LaCie, Amber, Uruk. Ping me when the push starts.

Congratulations on the full-band mixer ship — 9 aligned stems verified is the milestone this
whole thread was for.

— Pard
