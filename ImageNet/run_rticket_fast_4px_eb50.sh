# Remeber to change the path to the location of the ImageNet
DATA160=/datadrive_c/yucheng/imagenet-sz/160
DATA352=/datadrive_c/yucheng/imagenet-sz/352
DATA=/datadrive_c/yucheng/imagenet/

TICKET=/datadrive_c/yucheng/TLC/ResNet50

NAME=eps4

CONFIG1=configs_rticket/configs_fast_phase1_${NAME}.yml
CONFIG2=configs_rticket/configs_fast_phase2_${NAME}.yml
CONFIG3=configs_rticket/configs_fast_phase3_${NAME}.yml

PREFIX1=eb50_fast_adv_phase1_${NAME}
PREFIX2=eb50_fast_adv_phase2_${NAME}
PREFIX3=eb50_fast_adv_phase3_${NAME}

OUT1=eb50_fast_adv_phase1_${NAME}.out
OUT2=eb50_fast_adv_phase2_${NAME}.out
OUT3=eb50_fast_adv_phase3_${NAME}.out

EVAL1=eb50_fast_eval_phase1_${NAME}.out
EVAL2=eb50_fast_eval_phase2_${NAME}.out
EVAL3=eb50_fast_eval_phase3_${NAME}.out

END1=trained_models/eb50_fast_adv_phase1_${NAME}_step2_eps4_repeat1/checkpoint_epoch6.pth.tar
END2=trained_models/eb50_fast_adv_phase2_${NAME}_step2_eps4_repeat1/checkpoint_epoch12.pth.tar
END3=trained_models/eb50_fast_adv_phase3_${NAME}_step2_eps4_repeat1/checkpoint_epoch15.pth.tar

# training for phase 1
python -u main_fast_ticket.py $DATA160 -c $CONFIG1 --output_prefix $PREFIX1 --eb_path $TICKET/pruned_5008_0.5/pruned.pth.tar | tee $OUT1

# evaluation for phase 1
# python -u main_fast.py $DATA160 -c $CONFIG1 --output_prefix $PREFIX1 --resume $END1  --evaluate --eb_path $TICKET/pruned_5008_0.5/pruned.pth.tar | tee $EVAL1

# training for phase 2
python -u main_fast_ticket.py $DATA352 -c $CONFIG2 --output_prefix $PREFIX2 --eb_path $TICKET/pruned_5008_0.5/pruned.pth.tar  --resume $END1 | tee $OUT2

# evaluation for phase 2
# python -u main_fast.py $DATA352 -c $CONFIG2 --output_prefix $PREFIX2 --resume $END2 --evaluate --eb_path $TICKET/pruned_5008_0.5/pruned.pth.tar | tee $EVAL2

# training for phase 3
python -u main_fast_ticket.py $DATA -c $CONFIG3 --output_prefix $PREFIX3 --eb_path $TICKET/pruned_5008_0.5/pruned.pth.tar  --resume $END2 | tee $OUT3

# evaluation for phase 3
# python -u main_fast.py $DATA -c $CONFIG3 --output_prefix $PREFIX3 --resume $END3 --evaluate --eb_path $TICKET/pruned_5008_0.5/pruned.pth.tar | tee $EVAL3
