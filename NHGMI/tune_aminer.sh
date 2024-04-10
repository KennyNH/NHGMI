pos_list=$(cat tune_params/num_pos.list)

for num_pos in $pos_list
do
    python3 main.py aminer --patience 30 --verbose 1 --num_pos $num_pos | tee -a tune_params/aminer0402_5.log 
done