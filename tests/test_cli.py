import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from matrix_hmc_track import cli


class TestCliModelDiscovery(unittest.TestCase):
    def test_list_models_discovers_python_files_in_models_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            for name in ["alpha.py", "beta.py", "base.py", "utils.py", "_private.py"]:
                (models_dir / name).write_text("", encoding="utf-8")

            with patch.object(cli, "_MODEL_DIR", models_dir):
                discovered = cli._discover_known_models()
                with patch.object(cli, "_KNOWN_MODELS", discovered):
                    buffer = io.StringIO()
                    with contextlib.redirect_stdout(buffer):
                        with self.assertRaises(SystemExit) as exc_info:
                            cli.parse_args(["--list-models"])

        self.assertEqual(exc_info.exception.code, 0)

        output = buffer.getvalue()
        self.assertIn("alpha", output)
        self.assertIn("beta", output)
        self.assertNotIn("base", output)
        self.assertNotIn("utils", output)
        self.assertNotIn("_private", output)

    def test_complex64_alias_sets_precision(self) -> None:
        args = cli.parse_args(["--model", "yangmills", "--complex64"])
        self.assertEqual(args.precision, "complex64")

    def test_precision_flag_can_override_complex64_alias(self) -> None:
        args = cli.parse_args(["--model", "yangmills", "--complex64", "--precision", "complex128"])
        self.assertEqual(args.precision, "complex128")


if __name__ == "__main__":
    unittest.main()