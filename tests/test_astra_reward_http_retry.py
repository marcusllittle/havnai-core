"""Disposable coordinator HTTP acceptance; no production credentials or credits."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap


def test_real_reward_route_retry_preserves_exactly_one_credit(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = {key: value for key, value in os.environ.items()
           if key.upper() in {"SYSTEMROOT", "WINDIR", "PATH", "PYTHONPATH", "TEMP", "TMP"}}
    env.update({"PYTHON_DOTENV_DISABLED": "1", "HAVNAI_HOME": str(tmp_path),
                "HAVNAI_DB_PATH": str(tmp_path / "ledger.sqlite3"),
                "HAVNAI_NODES_PATH": str(tmp_path / "nodes.json")})
    for name, folder in {"STATIC": "static", "INSTALLER": "installers", "MANIFEST": "manifests",
                         "LOG": "logs", "OUTPUTS": "outputs", "ASSETS": "assets",
                         "MODEL_STORAGE": "models", "LORA_STORAGE": "loras"}.items():
        directory = tmp_path / folder
        directory.mkdir()
        env[f"HAVNAI_{name}_DIR"] = str(directory)
    (tmp_path / "manifests" / "registry.json").write_text('{"models": []}')
    code = textwrap.dedent('''
        import sys, time
        sys.path[:0] = [sys.argv[1], sys.argv[1] + "/server"]
        import app, astra_rewards, credits
        wallet = "0x" + "1" * 40
        client = app.app.test_client()
        with app.app.app_context():
            session = astra_rewards.create_session(wallet, 3600)["token"]
        headers = {"Authorization": "Bearer " + session}
        def open_run():
            response = client.post("/astra/run/start", json={"map_id": "nebula-runway"}, headers=headers)
            assert response.status_code == 200, response.json
            with app.app.app_context():
                db = app.get_db()
                db.execute("UPDATE astra_run_tokens SET started_at=? WHERE consumed_at IS NULL", (time.time()-300,))
                db.commit()
            return response.json["run_token"]
        payload = {"map_id": "nebula-runway", "score": 100000, "grade": "S", "duration_s": 999}
        payload["run_token"] = open_run()
        # Failure before settlement: authentication rejects the request.
        assert client.post("/astra/reward", json=payload).status_code == 401
        with app.app.app_context():
            assert credits.get_credit_balance(wallet) == 0
        successful = client.post("/astra/reward", json=payload, headers=headers)
        assert successful.status_code == 200, successful.json
        amount = successful.json["reward"]
        # Simulate loss of the successful response: resend the original payload.
        duplicate = client.post("/astra/reward", json=payload, headers=headers)
        assert duplicate.status_code == 422, duplicate.json
        assert duplicate.json["reason"] == "duplicate_run"
        assert duplicate.json["reward"] == 0
        with app.app.app_context():
            db = app.get_db()
            assert credits.get_credit_balance(wallet) == amount
            assert db.execute("SELECT COUNT(*) FROM astra_runs WHERE wallet=?", (wallet,)).fetchone()[0] == 1
            assert db.execute("SELECT COUNT(*) FROM credit_ledger WHERE wallet=? AND event_type='reward'", (wallet,)).fetchone()[0] == 1
            assert db.execute("SELECT total_deposited FROM credits WHERE wallet=?", (wallet,)).fetchone()[0] == amount
        print("HTTP_RETRY_EXACTLY_ONCE_OK")
    ''')
    result = subprocess.run([sys.executable, "-B", "-c", code, str(root)], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "HTTP_RETRY_EXACTLY_ONCE_OK" in result.stdout
