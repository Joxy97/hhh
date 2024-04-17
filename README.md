INSTALLATION:

git clone git@github.com:Joxy97/hhh.git
pip3 install -r requirements.txt

GET DATASETS:

scp /eos/cms/store/group/phys_higgs/cmshhh/v33/{path_to_dataset}.root ./data/ROOT_files/

CONVERT TO HDF5 FILE:

python3 -m src.convert_to_h5 data/ROOT_files/{signal}.root data/ROOT_files/{background_1}.root data/ROOT_files/{background_2}.root --out-file data/{name}_training.h5
python3 -m src.convert_to_h5 data/ROOT_files/{signal}.root data/ROOT_files/{background_1}.root data/ROOT_files/{background_2}.root --out-file data/{name}_testing.h5

PREPARE TRAINING:

Check .h5 file:             python3 -m src.scripts.print_h5 {path_to_h5_file}.h5
Setup events info file:     event_files/{event_info_file}.yaml
Setup training options:     options_files/{options_file}.json

TRAINING:

python3 -m spanet.train -of options_files/{options_file}.json --gpus {number}

EXPORT TO ONNX:

python3 -m spanet.export spanet_output/version_{number} spanet_output/version_{number}/spanet.onnx

INFERENCE:

Check .onnx file:           python3 -m src.scripts.print_onnx {path_to_onnx_file}.onnx
Change inference.py:        setup manually things in "class TestLightning(L.LightningModule)"
Run inference:              python3 -m inference spanet_output/version_{number}/spanet.onnx -tf data/{name}_testing.h5 --gpus {number}