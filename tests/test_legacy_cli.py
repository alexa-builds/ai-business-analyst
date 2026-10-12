import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_SCRIPT = PROJECT_ROOT / "main.py"


class LegacyCliTests(unittest.TestCase):
    def run_cli(self, csv_text):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "input.csv").write_text(csv_text, encoding="utf-8")
            environment = os.environ.copy()
            environment["MPLBACKEND"] = "Agg"
            environment["MPLCONFIGDIR"] = str(Path(directory, "matplotlib-config"))
            result = subprocess.run(
                [sys.executable, str(MAIN_SCRIPT)],
                input="input.csv\n",
                text=True,
                capture_output=True,
                cwd=directory,
                env=environment,
                timeout=30,
                check=False,
            )
            chart_exists = Path(directory, "sales_chart.png").exists()
        return result, chart_exists

    def test_single_zero_row_completes_without_division_by_zero(self):
        result, chart_exists = self.run_cli("month,sales\nJanuary,0\n")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("N/A (first sales value is zero)", result.stdout)
        self.assertIn("N/A (total sales are not positive)", result.stdout)
        self.assertTrue(chart_exists)

    def test_zero_start_with_later_growth_reports_change_as_unavailable(self):
        result, _ = self.run_cli("month,sales\nJanuary,0\nFebruary,10\n")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Change unavailable because starting sales are zero", result.stdout)

    def test_empty_header_is_reported_without_a_traceback(self):
        result, _ = self.run_cli("\n")

        self.assertEqual(result.returncode, 1)
        self.assertIn("CSV must contain 'month' and 'sales' columns", result.stdout)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
