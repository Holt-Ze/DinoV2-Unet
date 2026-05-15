import os
import tempfile
import unittest

from PIL import Image

from seg.checkpoints import clean_state_dict


def _write_image(path, mode="RGB"):
    image = Image.new(mode, (16, 16), color=255)
    image.save(path)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        try:
            from seg.data import KvasirSEG, _do_split
        except ModuleNotFoundError as exc:
            self.skipTest(f"dataset dependencies are not installed: {exc}")
        self.KvasirSEG = KvasirSEG
        self.do_split = _do_split

    def test_standard_split_and_kfold_lengths(self):
        names = [f"{idx:02d}.png" for idx in range(10)]
        self.assertEqual(len(self.do_split(names, "train", 0, 1)), 8)
        self.assertEqual(len(self.do_split(names, "val", 0, 1)), 1)
        self.assertEqual(len(self.do_split(names, "test", 0, 1)), 1)
        self.assertEqual(len(self.do_split(names, "train", 0, 5)), 7)
        self.assertEqual(len(self.do_split(names, "val", 0, 5)), 1)
        self.assertEqual(len(self.do_split(names, "test", 0, 5)), 2)

    def test_dataset_loads_mask_by_stem(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_dir = os.path.join(tmpdir, "images")
            mask_dir = os.path.join(tmpdir, "masks")
            os.makedirs(image_dir)
            os.makedirs(mask_dir)

            for idx in range(5):
                _write_image(os.path.join(image_dir, f"sample_{idx}.jpg"), "RGB")
                _write_image(os.path.join(mask_dir, f"sample_{idx}.png"), "L")

            dataset = self.KvasirSEG(tmpdir, split="train", img_size=32, aug_mode="none")
            image, mask, name = dataset[0]
            self.assertEqual(tuple(image.shape), (3, 32, 32))
            self.assertEqual(tuple(mask.shape), (1, 32, 32))
            self.assertTrue(name.endswith(".jpg"))


class CheckpointTests(unittest.TestCase):
    def test_clean_state_dict_removes_inference_incompatible_keys(self):
        state = {
            "module.encoder.weight": 1,
            "aux_heads.0.weight": 2,
            "decoder.total_ops": 3,
            "decoder.total_params": 4,
            "decoder.weight": 5,
        }
        cleaned = clean_state_dict(state)
        self.assertEqual(set(cleaned), {"encoder.weight", "decoder.weight"})


class AblationTests(unittest.TestCase):
    def test_ablation_config_maps_to_train_args(self):
        from run_ablation_studies import AblationScheduler, parse_args

        args = parse_args(
            [
                "--dataset",
                "kvasir",
                "--data-dir",
                "data/Kvasir-SEG",
                "--axes",
                "optimizer_strategy",
                "decoder_type",
                "pretrained_type",
            ]
        )
        cmd = AblationScheduler().config_to_args(
            {
                "optimizer_strategy": "frozen_encoder",
                "decoder_type": "complex",
                "pretrained_type": "imagenet_supervised",
            },
            args,
            save_dir_override="runs/ablation_runs/run_0001",
        )
        self.assertIn("--optimizer-strategy", cmd)
        self.assertIn("frozen_encoder", cmd)
        self.assertIn("--decoder-type", cmd)
        self.assertIn("complex", cmd)
        self.assertIn("--pretrained-type", cmd)
        self.assertIn("imagenet_supervised", cmd)
        self.assertIn("--no-export", cmd)


class CliParseTests(unittest.TestCase):
    def test_train_test_and_ablation_parse_smoke(self):
        import train
        import test
        from run_ablation_studies import parse_args as parse_ablation_args

        train_args = train.parse_args(["--dataset", "kvasir", "--data-dir", "data/Kvasir-SEG"])
        self.assertEqual(train_args.datasets, ["kvasir"])

        test_args = test.parse_args(
            [
                "--dataset",
                "kvasir",
                "--data-dir",
                "data/Kvasir-SEG",
                "--checkpoint",
                "runs/model/best.pt",
            ]
        )
        self.assertEqual(test_args.dataset, "kvasir")

        ablation_args = parse_ablation_args(
            [
                "--dataset",
                "kvasir",
                "--data-dir",
                "data/Kvasir-SEG",
                "--axes",
                "freeze_blocks_until",
                "lr_ratio",
                "--freeze-blocks-until",
                "6",
                "--lr-ratio",
                "0.01",
            ]
        )
        self.assertEqual(ablation_args.axes, ["freeze_blocks_until", "lr_ratio"])


if __name__ == "__main__":
    unittest.main()
