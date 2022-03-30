# Horizontal targets
for n in {1..3} ; do
    for i in {0..8} ; do
        for j in {0..8} ; do
            python3 src/synergetic/optimize_trajectory.py --task horizontal --n-synergies-1 ${i} --n-synergies-2 ${j} --out result/optim_h${n}/result_${i}_${j}
        done
    done
done

# Sagittal targets
for n in {1..3} ; do
    for i in {0..8} ; do
        for j in {0..8} ; do
            python3 src/synergetic/optimize_trajectory.py --task vertical --n-synergies-1 ${i} --n-synergies-2 ${j} --out result/optim_s${n}/result_${i}_${j}
        done
    done
done

# Frontal targets
for n in {1..3} ; do
    for i in {0..8} ; do
        for j in {0..8} ; do
            python3 src/synergetic/optimize_trajectory.py --task facing --n-synergies-1 ${i} --n-synergies-2 ${j} --out result/optim_f${n}/result_${i}_${j}
        done
    done
done

# Upper targets
for n in {1..3} ; do
    for i in {0..8} ; do
        for j in {0..8} ; do
            python3 src/synergetic/optimize_trajectory.py --task upper --n-synergies-1 ${i} --n-synergies-2 ${j} --out result/optim_u${n}/result_${i}_${j}
        done
    done
done
