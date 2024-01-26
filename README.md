# COOLER
[GCPR'23 Oral] COOLer: Class-Incremental Learning for Appearance-Based Multiple Object Tracking

## Installation

```bash
python3 -m pip install -r requirements.txt
python3 setup.py develop
```

Note that this project depends on other projects:

- [scalabel](https://github.com/scalabel/scalabel)
- [bdd100k](https://github.com/bdd100k/bdd100k)
 
The training framework in based on [Pytorch](https://pytorch.org/get-started/locally) and [PytorchLightning](https://www.pytorchlightning.ai/). And the detection models are based on [mmdet](https://github.com/open-mmlab/mmdetection). Please create a python environment and install specific versions of these packages suitable for your hardware.

Download the BDD100k tracking dataset to `data/bdd100k` and the SHIFT tracking dataset to `data/shift_dataset`.


## Running CL Experiments
Run BDD experiments:
```bash
bash run_scripts/run_bdd.sh  run_scripts/bdd_cfgs/most_to_least.toml --config configs/experiments/bdd/PL_GT_contrast/cooler_ct_loss.toml --exp_name most_to_least ## most to least setting
bash run_scripts/run_bdd.sh  run_scripts/bdd_cfgs/general_to_specific.toml --config configs/experiments/bdd/PL_GT_contrast/cooler_ct_loss.toml --exp_name general_to_specific ## general to specific setting
bash run_scripts/run_bdd.sh  run_scripts/bdd_cfgs/vehicle_bike_person.toml --config configs/experiments/bdd/PL_GT_contrast/cooler_ct_loss.toml --exp_name vehicle_bike_person ## vehicle to bike to human setting
```

Run SHIFT experiments:
```bash
bash run_scripts/run_shift.sh  run_scripts/shift_cfgs/most_to_least.toml --config configs/experiments/clear_daytime_shift/PL_GT_contrast/cooler_ct_loss.toml --exp_name most_to_least ## most to least setting
bash run_scripts/run_shift.sh  run_scripts/shift_cfgs/general_to_specific.toml --config configs/experiments/clear_daytime_shift/PL_GT_contrast/cooler_ct_loss.toml --exp_name general_to_sepcific ## general to specific setting
bash run_scripts/run_shift.sh  run_scripts/shift_cfgs/vehicle_bike_person.toml --config configs/experiments/clear_daytime_shift/PL_GT_contrast/cooler_ct_loss.toml --exp_name vehicle_bike_person ## vehicle to bike to human setting
```
## Usage


### Training

```bash
python3 -m ilmot.trainer train --config <config_path> <maybe other arguments>
```

### Testing

```bash
python3 -m ilmot.trainer test --config <config_path> <maybe other arguments>
```

### Prediction

```bash
python3 -m ilmot.trainer predict --config <config_path> <maybe other arguments>
```

### Visualization

```bash
python3 -m ilmot.trainer visualize --config <config_path> <maybe other arguments>
```
