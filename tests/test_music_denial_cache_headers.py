import ast
import unittest
from pathlib import Path


class MusicDenialCacheHeaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = Path("server/app.py").read_text(encoding="utf-8")
        cls.module = ast.parse(cls.source)

    def _function(self, name: str) -> ast.FunctionDef:
        for node in self.module.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
        self.fail(f"{name} not found")

    def test_publication_denials_use_no_store_helper(self) -> None:
        self.assertIn("def _no_store_json", self.source)
        for function_name in ("api_music_audio", "api_music_cover"):
            function_source = ast.get_source_segment(self.source, self._function(function_name)) or ""
            self.assertIn("_no_store_json", function_source)

    def test_cover_alias_uses_same_handler(self) -> None:
        route_paths = []
        for decorator in self._function("api_music_cover").decorator_list:
            if not (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == "route"
                and decorator.args
                and isinstance(decorator.args[0], ast.Constant)
            ):
                continue
            route_paths.append(decorator.args[0].value)

        self.assertIn("/music/publications/<publication_id>/cover.svg", route_paths)
        self.assertIn("/music/publications/<publication_id>/cover", route_paths)


if __name__ == "__main__":
    unittest.main()
