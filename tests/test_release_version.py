import ast
import subprocess
import tempfile
import unittest
from pathlib import Path


def load_resolve_version():
    source = Path("server/app.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    function = next(
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "resolve_version"
    )
    namespace = {"subprocess": subprocess}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "server/app.py", "exec"), namespace)
    return namespace["resolve_version"]


class ReleaseVersionTests(unittest.TestCase):
    def test_release_sha_file_is_used_for_archived_deploy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "RELEASE_SHA").write_text("3c58d04105bcc71a51c99d8696e05ea7445777f3\n", encoding="utf-8")
            resolve_version = load_resolve_version()
            resolve_version.__globals__.update({
                "BASE_DIR": base,
                "VERSION_FILE": base / "VERSION",
            })

            self.assertEqual(resolve_version(), "3c58d04105bc")

    def test_node_version_file_still_takes_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "VERSION").write_text("node-version-123\n", encoding="utf-8")
            (base / "RELEASE_SHA").write_text("3c58d04105bcc71a51c99d8696e05ea7445777f3\n", encoding="utf-8")
            resolve_version = load_resolve_version()
            resolve_version.__globals__.update({
                "BASE_DIR": base,
                "VERSION_FILE": base / "VERSION",
            })

            self.assertEqual(resolve_version(), "node-version-123")


if __name__ == "__main__":
    unittest.main()
