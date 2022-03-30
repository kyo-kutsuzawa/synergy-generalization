# Frontal targets
for n in {1..3} ; do
    for i in {0..8} ; do
        for j in {0..8} ; do
            python3 src/synergetic/optimize_trajectory.py --task facing --arm short --n-synergies-1 ${i} --n-synergies-2 ${j} --out result/optim_f_short${n}/result_${i}_${j} &
            python3 src/synergetic/optimize_trajectory.py --task facing --arm long  --n-synergies-1 ${i} --n-synergies-2 ${j} --out result/optim_f_long${n}/result_${i}_${j}
        done
    done
done
