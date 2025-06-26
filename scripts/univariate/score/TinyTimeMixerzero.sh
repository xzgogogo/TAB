python ./scripts/run_benchmark.py --config-path "unfixed_detect_score_config.json" --model-name "pre_train.TinyTimeMixer" --model-hyper-params '{"horizon": 1, "norm": true, "num_epochs": 3, "seq_len": 96}' --adapter "PreTrain_adapter" --gpus 0 --num-workers 1 --timeout 60000 --save-path "score/TinyTimeMixerzero"

