#!/usr/bin/env bash
# Final synchronization only when neither organizer worker is writing.
set -euo pipefail
cd /root/nuhack
host=ubuntu@global.prd.ga.run.brev.nvidia.com
common=(-i /root/.ssh/nuhack_brev -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=10)
for port in 21453 15171; do
  if ssh "${common[@]}" -p "$port" "$host" "pgrep -af '[p]ython.*(organizer.analyze|t4_organizer.py)'"; then
    echo "Organizer still running on port $port; final sync deferred." >&2
    exit 75
  else
    rc=$?
    if [[ "$rc" != 1 ]]; then echo 'Cannot verify worker state; synchronization stopped.' >&2; exit "$rc"; fi
  fi
done
mkdir -p backups/migration_20261017/final_organizer_v1
rsync -az --exclude env -e "ssh -i /root/.ssh/nuhack_brev -p 21453 -o BatchMode=yes -o IdentitiesOnly=yes" "$host:/home/ubuntu/nuhack-training/runs/narration_training/organizer_v1/" backups/migration_20261017/final_organizer_v1/
rsync -az -e "ssh -i /root/.ssh/nuhack_brev -p 15171 -o BatchMode=yes -o IdentitiesOnly=yes" backups/migration_20261017/final_organizer_v1/ "$host:/home/ubuntu/nuhack-training/runs/narration_training/organizer_v1/"
echo 'Final progress synchronized. Resume on T4 using HANDOFF.txt.'
