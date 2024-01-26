"""Continual learning protocal for calling vis4d in multiple stages."""
import os
import copy
import sys

from ilmot.trainer import run_stage, get_args


def main():
    """Main function."""
    # fmt: off
    cfg, trainer_args, run_cfg = get_args(use_run_cfg=True) # pylint: disable=unbalanced-tuple-unpacking
    # fmt: on
    class_order = run_cfg["class_order"]
    # for reproducible results
    cfg.launch.seed = 777
    exp_name = cfg.launch.exp_name
    stage = ""
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
            for new_class in new_classes:
                stage = stage + "_" + new_class
            continue

        ## predict pseudo labels
        cfg.launch.resume = False
        cfg.model["train_category"] = new_classes
        for new_class in new_classes:
            cfg.model["test_category"].append(new_class)
            cfg.model["category_mapping"][new_class] = len(
                cfg.model["category_mapping"]
            )
        new_model_path = os.path.join(
            "ilmot-workspace", exp_name, stage, "checkpoints/last.ckpt"
        )
        cfg.launch.weights = new_model_path
        cfg.launch.version = stage + "_pl"
        cfg.ignore_new_class = True
        cfg.load_vist = False
        cfg.launch.action = "predict"
        print(f"predicting pseudo labels for: {stage}")
        run_stage(cfg, copy.deepcopy(trainer_args))
        sys.exit(0)


if __name__ == "__main__":  # pragma: no cover
    main()
    sys.exit(1)
