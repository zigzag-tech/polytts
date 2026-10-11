"""CPU source wiring controls; no full-server/model qualification."""
import ast
import asyncio
from pathlib import Path
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'server.py'

class QualificationStartupTests(unittest.TestCase):
    def test_qualified_startup_has_no_effects(self):
        tree = ast.parse(SOURCE.read_text())
        startup = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == 'startup')
        startup.decorator_list = []
        namespace = {"QUALIFICATION_ONLY": True}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[startup], type_ignores=[])), str(SOURCE), "exec"), namespace)
        asyncio.run(namespace[startup.name]())

    def test_default_startup_retains_original_body(self):
        tree = ast.parse(SOURCE.read_text())
        startup = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == 'startup')
        guard = next(node for node in startup.body if isinstance(node, ast.If))
        self.assertEqual(ast.dump(guard.test), "Name(id='QUALIFICATION_ONLY', ctx=Load())")
        self.assertEqual(len(guard.body), 1)
        self.assertIsInstance(guard.body[0], ast.Return)
        self.assertGreater(len(startup.body), 3)

if __name__ == "__main__":
    unittest.main()
