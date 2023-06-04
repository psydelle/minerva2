echo "en_en start";
python testing_parallel.py -l en -f en -n 230 --num_workers 8 --minerva_k 0.97 --minerva_max_iter 700;
echo "en_en done";

echo "en_pt start";
python testing_parallel.py -l en -f pt -n 230 --num_workers 8 --minerva_k 0.97 --minerva_max_iter 700;
echo "en_pt done";

echo "en_mix start";
python testing_parallel.py -l en -f mix -n 230 --num_workers 8 --minerva_k 0.97 --minerva_max_iter 700;
echo "en_mix done";

echo "pt_en start";
python testing_parallel.py -l pt -f en -n 230 --num_workers 8 --minerva_k 0.97 --minerva_max_iter 700;
echo "pt_en done";

echo "pt_pt start";
python testing_parallel.py -l pt -f pt -n 230 --num_workers 8 --minerva_k 0.97 --minerva_max_iter 700;
echo "pt_pt done";

echo "pt_mix start";
python testing_parallel.py -l pt -f mix -n 230 --num_workers 8 --minerva_k 0.97 --minerva_max_iter 700;
echo "pt_mix done";

# echo "en_aligned_en start";
# python testing_parallel.py -l en_aligned -f en -n 230 --num_workers 8 --minerva_k 0.97 --minerva_max_iter 700 --en_pt_trans_pickle stimuli_en_to_pt-concat.pkl;
# echo "en_aligned_en done";

# echo "en_aligned_pt start";
# python testing_parallel.py -l en_aligned -f pt -n 230 --num_workers 8 --minerva_k 0.97 --minerva_max_iter 700 --en_pt_trans_pickle stimuli_en_to_pt-concat.pkl;
# echo "en_aligned_pt done";

# echo "en_aligned_mix start"
# python testing_parallel.py -l en_aligned -f mix -n 230 --num_workers 8 --minerva_k 0.97 --minerva_max_iter 700 --en_pt_trans_pickle stimuli_en_to_pt-concat.pkl;
# echo "en_aligned_mix done";

# 4w
# real    1m3.045s
# user    2m13.880s
# sys     1m42.023s

# 8w
# real    1m0.058s
# user    2m58.004s
# sys     2m38.754s

# 4w-thread
# real    0m58.723s
# user    1m47.046s
# sys     2m36.314s

# 8w-thread
# real    0m51.771s
# user    1m48.376s
# sys     2m30.414s

# 16w-thread
# real    0m49.647s
# user    1m53.571s
# sys     2m36.865s

# 32w-thread
# real    0m49.661s
# user    1m56.904s
# sys     2m40.080s
