evaluation

CUDA_VISIBLE_DEVICES=0 python train_gbgdn_detected.py --id 'GBGDN' --batch_size 16 --model_method 'GBGDN' --evaluate True --model ./output/checkpoints/gbgdn/model_best.pth.tar


training
CUDA_VISIBLE_DEVICES=0 python train_gbgdn_detected.py --learning_rate 1e-4 --id 'GBGDN' --word_drop_out 0.2 --rnn_drop_out 0.2 --jemb_drop_out 0.2 --batch_size 16 --edge_gate_drop_out 0.0 --word_judge_drop 0.0 --model_method 'GBGDN' --max_epochs 60


