import pandas as pd
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack


def _clean_data(data):
    data.loc[data["Payee"].str.startswith("Transfer :"), "Category Group/Category"] = (
        "Transfer"
    )

    data["Payee"] = data["Payee"].fillna("")
    data["Category Group/Category"] = data["Category Group/Category"].fillna(
        "Uncategorized"
    )
    counts = data["Category Group/Category"].value_counts()
    keep = counts[counts >= 10].index
    data = data[data["Category Group/Category"].isin(keep)]

    payee_vectorizer = TfidfVectorizer()
    payee_features = payee_vectorizer.fit_transform(data["Payee"])
    data["Outflow"] = data["Outflow"].replace(r"[\$,]", "", regex=True).astype(float)
    data["Inflow"] = data["Inflow"].replace(r"[\$,]", "", regex=True).astype(float)

    money_features = data[["Outflow", "Inflow"]].values

    features = hstack([payee_features, money_features])

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(data["Category"])

    return features, labels


raw = pd.read_csv("test_data/rh-ynab_test_data.csv")

data, labels = _clean_data(raw)

# Split
traindata, testdata, trainlabels, testlabels = train_test_split(
    data, labels, test_size=0.2
)

# Train
# Set num_class explicitly so XGBoost doesn't fail when train/test splits have different classes
model = xgboost.XGBClassifier(num_class=len(set(labels)))
model.fit(traindata, trainlabels)

# Evaluate
print(model.score(testdata, testlabels))
