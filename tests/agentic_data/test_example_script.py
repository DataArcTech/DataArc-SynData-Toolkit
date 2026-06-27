import unittest
from pathlib import Path


class SynAgenticExampleScriptTests(unittest.TestCase):
    def test_script_file_exists(self):
        self.assertTrue(Path("examples/syn_agentic_data/run.py").exists())


if __name__ == "__main__":
    unittest.main()
