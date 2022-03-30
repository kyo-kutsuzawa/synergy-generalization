# Horizontal policy
mkdir result/result_h_save
python3 src/rl/evaluate.py --model results/result_h_save/pyt_save/model10000.pt --norender --out result/result_h_save/result0.pickle
for i in {1..29} ; do
    python3 src/rl/evaluate.py --model results/result_h_save/pyt_save/model${i}010000.pt --norender --out result/result_h_save/result${i}0.pickle
done
python3 src/rl/evaluate.py --model results/result_h_save/pyt_save/model30000000.pt --norender --out result/result_h_save/result300.pickle


# Sagittal policy
mkdir result/result_v_save
python3 src/rl/evaluate.py --model results/result_v_save/pyt_save/model10000.pt --norender --out result/result_v_save/result0.pickle
for i in {1..29} ; do
    python3 src/rl/evaluate.py --model results/result_v_save/pyt_save/model${i}010000.pt --norender --out result/result_v_save/result${i}0.pickle
done
python3 src/rl/evaluate.py --model results/result_v_save/pyt_save/model30000000.pt --norender --out result/result_v_save/result300.pickle
