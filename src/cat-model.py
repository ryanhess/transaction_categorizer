import pandas as pd
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def clean_data(data):
    data.loc[data["Payee"].str.startswith("Transfer :"), "Category Group/Category"] = (
        "Transfer"
    )

    payee_encoder = LabelEncoder()
    data["Payee"] = payee_encoder.fit_transform(data["Payee"])
    data["Outflow"] = data["Outflow"].replace(r"[\$,]", "", regex=True).astype(float)
    data["Inflow"] = data["Inflow"].replace(r"[\$,]", "", regex=True).astype(float)

    features = data[["Payee", "Outflow", "Inflow"]]

    # label: Category Group/Category as ints
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(data["Category Group/Category"])

    return features, labels


# get the data from the ynab file
raw = pd.read_csv("test_data/rh-ynab_test_data.csv")

data, labels = clean_data(raw)

# Split
traindata, testdata, trainlabels, testlabels = train_test_split(
    data, labels, test_size=0.2
)

# Train
model = xgboost.XGBClassifier()
model.fit(traindata, trainlabels)

# Evaluate
print(model.score(testdata, testlabels))
