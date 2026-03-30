_path_to_model_state = "transaction_categorizer/inference/cat/state/"

training_data_filepath = (
    "transaction_categorizer/inference/cat/training_data/ynab-rh-txns.csv"
)
training_params_filepath = _path_to_model_state + "training_params.json"
model_filepath = _path_to_model_state + "model.json"
payee_vectorizer_filepath = _path_to_model_state + "payee_vectorizer.pkl"
label_encoder_filepath = _path_to_model_state + "category_encoder.pkl"
