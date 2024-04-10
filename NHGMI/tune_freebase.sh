pos_list=$(cat tune_params/num_pos_freebase.list)

for num_pos in $pos_list
do
    python3 main.py freebase --patience 30  --verbose 1 --num_pos $num_pos | tee -a tune_params/freebase0409.log 
done