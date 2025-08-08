$input_dir=/home/dan/uw_mortality_prediction_omop_data/evaluation_small_meds/data
$output_dir=/home/dan/uw_mortality_prediction_omop_data/evaluation_small_ethos

ethos_tokenize -m worker=1    input_dir=$input_dir/train   output_dir=$output_dir  out_fn=trainmkdir evaluation_small_ethos