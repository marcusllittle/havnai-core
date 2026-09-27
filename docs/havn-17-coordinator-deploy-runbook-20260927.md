# HAVN-17 Coordinator Deploy Runbook Correction - 2026-09-27

The live coordinator no longer uses the older `/opt/havnai/releases` layout.
Production evidence shows `havnai-coordinator.service` runs as `marcus` from:

```text
/home/marcus/Downloads/source-code/havnai-core
```

The deploy script now defaults to that live layout:

- coordinator SSH target: `marcus@192.168.4.105`
- coordinator checkout: `/home/marcus/Downloads/source-code/havnai-core`
- venv: `/home/marcus/Downloads/source-code/havnai-core/.venv`
- database: `/home/marcus/Downloads/source-code/havnai-core/db/ledger.db`
- deploy backup directory:
  `/home/marcus/Downloads/source-code/havnai-core/db/backups`
- service: `havnai-coordinator.service`

The coordinator deploy path now:

1. Records the previous git commit in `/tmp/havnai-$SHA.previous`.
2. Preserves runtime-managed `nodes.json`.
3. Fetches and fast-forwards the release branch checkout.
4. Restores `nodes.json`.
5. Installs requirements with the repo venv.
6. Runs a pre-restart SQLite backup using the live DB path.
7. Restarts `havnai-coordinator.service`.
8. Polls `http://127.0.0.1:5001/healthz`.

This change removes the stale assumptions that caused the production deploy
failure:

- `/opt/havnai/releases/$SHA`
- `/opt/havnai/venv`
- `/var/lib/havnai/ledger.db`
- Unix user/group `havnai:havnai`

Rollback support is still limited to the coordinator git checkout and service
restart. HAVN-46 remains open until coordinator and node rollback drills are
exercised or waived.
