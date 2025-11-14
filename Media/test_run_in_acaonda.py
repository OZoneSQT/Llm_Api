import subprocess
import shutil
import sys
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock

import run_in_acaonda as ria


class TestRunInAcaonda(unittest.TestCase):
    @patch("shutil.which", return_value="/usr/bin/conda")
    @patch("subprocess.run")
    def test_successful_run(self, mock_run: MagicMock, _mock_which: MagicMock):
        mock_run.return_value = subprocess.CompletedProcess(args=["conda"], returncode=0)
        rc = ria.run_in_acaonda(Path("dummy.py"), ["--foo"])
        self.assertEqual(rc, 0)
        mock_run.assert_called()

    @patch("shutil.which", return_value=None)
    def test_conda_missing(self, _mock_which: MagicMock):
        rc = ria.run_in_acaonda(Path("dummy.py"), [])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
