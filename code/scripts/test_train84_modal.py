"""Tests for the Modal launcher of the 84-field trainers (issue #431, scripts/train84_modal.py).

Clean-machine safe except the laptop-only class, which checks the launcher against the trainers and this laptop's inputs.

    python scripts/test_train84_modal.py   (from code/)
"""
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import repo_paths  # noqa: F401
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train84_modal as TM

LAPTOP = Path("/tmp/vgc-pilot/data/pokedex.json").exists()


class NamesAndCommandsTests(unittest.TestCase):
    def test_run_names(self):
        self.assertEqual(TM.run_name("udlm_loglinear_gidd_lr"), "stat84_udlm_loglinear_lr5e4_min5e5_es3_cap80")
        self.assertEqual(TM.run_name("shuffled_linear_filtered_gidd_lr", smoke=True),
                         "stat84smoke_stat84_shuffled_noise_linear_filtered_lr5e4_min5e5_es3_cap80")
        with self.assertRaises(ValueError):
            TM.run_name("udlm_loglinear")

    def test_command_runs_the_trainer_cli_through_boot(self):
        cmd = TM.trainer_command("shuffled_linear_filtered_gidd_lr", legality_teams=64)
        self.assertEqual(cmd[1:3], ["-u", "-c"])
        self.assertEqual(cmd[3], TM.BOOT)
        self.assertEqual(cmd[4:], [f"{TM.REMOTE_CODE}/src/train84.py", "shuffled_linear_filtered_gidd_lr",
                                   "--legality-teams", "64", "--wandb-mode", "offline"])

    def test_legality_command_runs_checkpoint_legality_through_boot(self):
        cmd = TM.legality_command("stat84_shuffled_noise_linear_filtered_es3_cap80")
        self.assertEqual(cmd[3], TM.BOOT)
        self.assertEqual(cmd[4:], [f"{TM.REMOTE_CODE}/src/checkpoint_legality.py",
                                   "stat84_shuffled_noise_linear_filtered_es3_cap80"])

    def test_legality_command_passes_guidance(self):
        self.assertEqual(TM.legality_command("stat84_shuffled_noise_linear_filtered_es3_cap80", 1.0)[-2:],
                         ["--guidance", "1"])

    def test_smoke_flag_is_the_trainers_env_switch(self):
        self.assertEqual(TM.trainer_env(smoke=True, base={})["S84_SMOKE"], "1")
        self.assertNotIn("S84_SMOKE", TM.trainer_env(smoke=False, base={"S84_SMOKE": "1"}))
        self.assertEqual(TM.trainer_env(base={})["WANDB_DIR"], "/vol/wandb")

    def test_checkpoints_folder_points_at_the_volume(self):
        self.assertIn((TM.CHECKPOINTS, Path("/vol/lossdown")), TM.symlinks())

    def test_gpus_support_bf16(self):
        self.assertNotIn("T4", TM.GPU)

    def test_fetch_commands_name_the_checkpoint_and_epoch_folder(self):
        commands = TM.fetch_commands("udlm_loglinear_gidd_lr")
        self.assertTrue(any("lossdown/stat84_udlm_loglinear_lr5e4_min5e5_es3_cap80.pt" in c for c in commands))
        self.assertTrue(any(c.split()[4] == "lossdown/stat84_udlm_loglinear_lr5e4_min5e5_es3_cap80" for c in commands))


class BootTests(unittest.TestCase):
    def test_boot_sets_dev_to_cuda_on_load_and_runs_the_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "diffusion.py").write_text('DEV = "cpu"\n')
            Path(tmp, "copier.py").write_text("import diffusion as D\nDEV = D.DEV\n")
            script = Path(tmp, "trainer.py")
            script.write_text(textwrap.dedent("""
                import sys
                sys.path.insert(0, %r)
                import copier
                print("DEV", copier.DEV, "ARGV", sys.argv[1:])
            """ % tmp))
            env = dict(os.environ, TRAIN84_MODAL_REQUIRE_CUDA="0")
            out = subprocess.run([sys.executable, "-c", TM.BOOT, str(script), "arm", "--flag"], env=env,
                                 capture_output=True, text=True, check=True).stdout
        self.assertIn("DEV cuda ARGV ['arm', '--flag']", out)
        self.assertIn("[modal] diffusion.DEV -> cuda", out)


@unittest.skipUnless(LAPTOP, "needs this laptop's /tmp/vgc-pilot data")
class LaptopTests(unittest.TestCase):
    def test_every_input_exists(self):
        for local, _remote in TM.runtime_inputs():
            self.assertTrue(local.exists(), local)

    def test_checkpoint_names_match_the_trainers(self):
        sys.path.insert(1, "/tmp/vgc-pilot/src")
        import train84 as TR
        import train_uniform84 as TU
        self.assertEqual(TM.ARMS["udlm_loglinear_gidd_lr"]["checkpoint"], TU.NAMES["udlm_loglinear_gidd_lr"])
        self.assertEqual(TM.ARMS["shuffled_linear_filtered_gidd_lr"]["checkpoint"], TR.NAMES["shuffled_linear_filtered_gidd_lr"])
        self.assertEqual(TM.CHECKPOINTS, TR.CK)
        self.assertEqual(TM.RESULTS, TR.X.RES)

    def test_legality_result_name_matches_checkpoint_legality(self):
        import checkpoint_legality as CL  # its corpus import reads /tmp/vgc-pilot/data
        name = "stat84_shuffled_noise_linear_filtered_es3_cap80"
        self.assertEqual(TM.legality_result_name(name), CL.result_path(name, tables=False).name)
        self.assertEqual(TM.legality_result_name(name, 1.0), CL.result_path(name, tables=False, guidance=1.0).name)


if __name__ == "__main__":
    unittest.main()
