pos_list=$(cat tune_params/num_pos.list)

for num_pos in $pos_list
do
    python3 main.py imdb --patience 30 --num_pos $num_pos --verbose 1 | tee -a tune_params/imdb0408.log
done