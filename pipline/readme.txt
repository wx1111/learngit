train: python train_early_stop.py --pre_trained_model_ckpt_path --checkpoint_dir --train_data_dir --validation_data_dir --num_class --export_path_base --model_name

eval: python eval_model.py --data_dir --checkpoint_dir --output_dir

performence: python preformeance.py

image2tfrecord: python image2tfrecord.py --data_dir --output_dir --subset
 
