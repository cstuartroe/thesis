import json
from urllib import request as ur
import pandas as pd
from tqdm import tqdm


language_info = pd.read_csv("csv/language_info.csv")


FEATURES = ["20A", "21A", "21B", "22A", "23A", "24A", "25A", "25B", "26A", "27A", "28A", "29A"]


def fetch_json(url):
    raw = ur.urlopen(url).read().decode("utf-8")
    return json.loads(raw)


def pull_feature(feature):
    language_values = {}

    feature_info_url = f"https://wals.info/feature/{feature}.geojson"
    feature_info = fetch_json(feature_info_url)
    feature_name = feature_info["properties"]["name"]
    for feature_value in feature_info["properties"]["domain"]:
        feature_value_url = feature_info_url + f"?domainelement={feature_value['id']}&layer={feature_value['id']}"
        feature_value_info = fetch_json(feature_value_url)
        for language in feature_value_info["features"]:
            iso_codes = language["properties"]["language"]["iso_codes"].split(", ")
            value = language["properties"]["value_name"]
            for iso_code in iso_codes:
                language_values[iso_code] = value

    if feature_name in language_info.columns:
        language_info.drop(columns=[feature_name])

    feature_values = [language_values.get(iso_code, "-") for iso_code in language_info["ISO 639-2"]]
    language_info[feature_name] = feature_values


def pull():
    for feature in tqdm(FEATURES):
        pull_feature(feature)

    language_info.to_csv("csv/language_info.csv", index=False)
