DATA=/datadrive_c/yucheng/imagenet/


# 4eps evaluation
# EB 30
python main_fast_ticket.py $DATA --config configs_rticket/configs_fast_4px_evaluate.yml --output_prefix eval_eb30_4px --resume trained_models/eb30_fast_adv_phase3_eps2_step2_eps4_repeat1/model_best.pth.tar --evaluate --restarts 10 --eb_path $TICKET/pruned_3010_0.3/pruned.pth.tar
# EB 50
python main_fast_ticket.py $DATA --config configs_rticket/configs_fast_4px_evaluate.yml --output_prefix eval_eb50_4px --resume trained_models/eb50_fast_adv_phase3_eps2_step2_eps4_repeat1/model_best.pth.tar --evaluate --restarts 10 --eb_path $TICKET/pruned_5008_0.5/pruned.pth.tar
# EB 70
python main_fast_ticket.py $DATA --config configs_rticket/configs_fast_4px_evaluate.yml --output_prefix eval_eb70_4px --resume trained_models/eb70_fast_adv_phase3_eps2_step2_eps4_repeat1/model_best.pth.tar --evaluate --restarts 10 --eb_path $TICKET/pruned_7008_0.7/pruned.pth.tar


# 2eps evaluation
# EB 30
python main_fast_ticket.py $DATA --config configs_rticket/configs_fast_2px_evaluate.yml --output_prefix eval_eb30_2px --resume trained_models/eb30_fast_adv_phase3_eps2_step2_eps2_repeat1/model_best.pth.tar --evaluate --restarts 10 --eb_path $TICKET/pruned_3010_0.3/pruned.pth.tar
# EB 50
python main_fast_ticket.py $DATA --config configs_rticket/configs_fast_2px_evaluate.yml --output_prefix eval_eb50_2px --resume trained_models/eb50_fast_adv_phase3_eps2_step2_eps2_repeat1/model_best.pth.tar --evaluate --restarts 10 --eb_path $TICKET/pruned_5008_0.5/pruned.pth.tar
# EB 70
python main_fast_ticket.py $DATA --config configs_rticket/configs_fast_2px_evaluate.yml --output_prefix eval_eb70_2px --resume trained_models/eb70_fast_adv_phase3_eps2_step2_eps2_repeat1/model_best.pth.tar --evaluate --restarts 10 --eb_path $TICKET/pruned_7008_0.7/pruned.pth.tar
