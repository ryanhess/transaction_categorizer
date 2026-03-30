_path_to_model_state_str = "transaction_categorizer/inference/cat/state/"

training_data_filepath = (
    "transaction_categorizer/inference/cat/training_data/ynab-rh-txns.csv"
)

training_params_filepath = _path_to_model_state_str + "training_params.json"
model_filepath = _path_to_model_state_str + "model.json"
payee_vectorizer_filepath = _path_to_model_state_str + "payee_vectorizer.pkl"
label_encoder_filepath = _path_to_model_state_str + "category_encoder.pkl"
