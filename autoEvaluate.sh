
# argv :
#   $1  : inference data
#   $2  : features save path
#   $3  : Data type [ "all" , "pos" , "length" , "rhyme" ]

python3 metrics/gen_predict_features.py $1 $2
python3 metrics/evaluate.py $2 $3 $1