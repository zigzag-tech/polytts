"""CPU source wiring controls; no full-server/model qualification."""
import ast
import asyncio
import hashlib
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
        retained = [node for node in startup.body if node is not guard]
        module = ast.Module(body=retained, type_ignores=[])
        # Python3.12 adds an empty type_params field to ordinary definitions.
        # Preserve the exact3.11 baseline serialization; nonempty type params
        # remain in the dump and cannot pass the unchanged original hash.
        for node in ast.walk(module):
            if hasattr(node,'type_params') and node.type_params == []:
                delattr(node,'type_params')
        digest = hashlib.sha256(ast.dump(module, include_attributes=False).encode()).hexdigest()
        # Retained original startup AST before adding the qualification guard.
        self.assertEqual(digest, 'bb1a13f3d1c74fa36863e2dbe3fa181ea5d141f42e69fcd69a2915e466b7ddd8')


class PreloadTests(unittest.TestCase):
    def test_both_attach_preloads_are_disabled_only_for_qualification(self):
        tree = ast.parse(SOURCE.read_text())
        preloads = [kw.value for node in ast.walk(tree) if isinstance(node, ast.Call) for kw in node.keywords if kw.arg == "preload"]
        self.assertEqual(len(preloads), 2)
        for value in preloads:
            self.assertIsInstance(value, ast.IfExp)
            self.assertEqual(ast.dump(value.test), "Name(id='QUALIFICATION_ONLY', ctx=Load())")
            self.assertIsNone(value.body.value)
            self.assertIsInstance(value.orelse, ast.Lambda)

if __name__ == "__main__":
    unittest.main()
