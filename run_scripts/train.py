"""Continual learning protocal for calling vis4d in multiple stages."""
import os
import sys
import copy

from ilmot.trainer import run_stage, get_args


def main():
    """Main function."""
    # fmt: off
    cfg, trainer_args, run_cfg = get_args(use_run_cfg=True)  # pylint: disable=unbalanced-tuple-unpacking
    # fmt: on
    class_order = run_cfg["class_order"]
    # for reproducible results
    cfg.launch.seed = 777
    exp_name = cfg.launch.exp_name
    stage = ""
    prev_stage = ""
    for i, new_classes in enumerate(class_order):
        # go to next stage
        if i == 0 or os.path.exists(
            os.path.join(
                "ilmot-pseudolabel",
                exp_name,
                stage + "_pl",
                "track.pkl",
            )
        ):
            cfg.model["train_category"] = new_classes
            for new_class in new_classes:
                cfg.model["test_category"].append(new_class)
                cfg.model["category_mapping"][new_class] = len(
                    cfg.model["category_mapping"]
                )
            prev_stage = stage
            for new_class in new_classes:
                stage = stage + "_" + new_class
            continue

        # train on new class

        # resume
        cfg.ignore_new_class = False

        if i > 1:
            if cfg.use_pl_dataset:
                cfg.pl_dataset_path = os.path.join(
                    "ilmot-pseudolabel",
                    exp_name,
                    prev_stage + "_pl",
                    "track.pkl",
                )
            else:
                cfg.pl_dataset_path = None
        if os.path.exists(
            os.path.join(
                "ilmot-workspace",
                exp_name,
                stage,
                "checkpoints/last.ckpt",
            )
        ):
            cfg.launch.resume = True
            cfg.launch.weights = None

        else:
            cfg.launch.resume = False
            if i == 1:
                cfg.launch.weights = None
            else:
                cfg.load_vist = False
                old_model_path = os.path.join(
                    "ilmot-workspace",
                    exp_name,
                    prev_stage,
                    "checkpoints/last.ckpt",
                )
                cfg.launch.weights = old_model_path

        # cfg.model["disable_contrast"] = i == 1
        cfg.launch.version = stage
        print(f"training on: {stage}")
        cfg.launch.action = "train"
        run_stage(cfg, copy.deepcopy(trainer_args))
        sys.exit(0)


if __name__ == "__main__":  # pragma: no cover
    main()
    sys.exit(1)
