# python tests/test_lmze.py --path /mnt/nfs2/zhanghe/NuAgent/exp/LM-ready/baselinesd000_1733114808_lm_SZPC -e 0 --eval_type 'random'
# python tests/test_lmze.py --path /mnt/nfs2/zhanghe/NuAgent/exp/LM-ready/baselinesd000_1733114808_lm_SZPC -e 500 --eval_type 'random'


# python tests/test_lmze.py --path /mnt/nfs2/zhanghe/NuAgent/exp/LM-ready/Ours-OnlyRegretScale-lr_te_1e_3-wodsd000_1733208892_lm_SZPC -e 1800 --eval_type 'random_psi'


# python tests/test_lmze.py --path /mnt/nfs2/zhanghe/NuAgent/exp/LM-ready/baselinesd000_1733133625_lm_metra_bl -e 1800 --eval_type 'random_psi'

# python tests/test_lmze.py --path /mnt/nfs2/zhanghe/NuAgent/exp/LM-ready/Ours-w15r3sd008_1733717931_lm_SZPC -e 1000 --eval_type window_psi


# python tests/test_lmze.py --path /mnt/nfs2/zhanghe/NuAgent/exp/LM-ready/baselinesd004_1733241785_lm_metra_bl -e 1000 --eval_type random
python tests/viz_utils.py --eval_type random --model_path /mnt/nfs2/zhanghe/NuAgent/exp/ant/baselinesd000_1733299484_ant_metra_bl
python tests/viz_utils.py --eval_type random --model_path /mnt/nfs2/zhanghe/NuAgent/exp/ant/baselinesd002_1733416521_ant_metra_bl
python tests/viz_utils.py --eval_type random --model_path /mnt/nfs2/zhanghe/NuAgent/exp/ant/baselinesd004_1733438971_ant_metra_bl
python tests/viz_utils.py --eval_type random --model_path /mnt/nfs2/zhanghe/NuAgent/exp/ant/baselinesd008_1733460821_ant_metra_bl
python tests/viz_utils.py --eval_type random --model_path /mnt/nfs2/zhanghe/NuAgent/exp/ant/baselinesd016_1733482995_ant_metra_bl


python tests/viz_utils.py --eval_type random_psi --model_path /mnt/nfs2/zhanghe/NuAgent/exp/ant/Ours-Win20sd000_1733465312_ant_SZPC
python tests/viz_utils.py --eval_type random_psi --model_path /mnt/nfs2/zhanghe/NuAgent/exp/ant/Ours-Win20sd002_1733489607_ant_SZPC
python tests/viz_utils.py --eval_type random_psi --model_path /mnt/nfs2/zhanghe/NuAgent/exp/ant/Ours-Win20sd004_1733491014_ant_SZPC
python tests/viz_utils.py --eval_type random_psi --model_path /mnt/nfs2/zhanghe/NuAgent/exp/ant/Ours-Win20sd008_1733516784_ant_SZPC
python tests/viz_utils.py --eval_type random_psi --model_path /mnt/nfs2/zhanghe/NuAgent/exp/ant/Ours-Win20sd016_1733541990_ant_SZPC


python PlotPickle.py 

