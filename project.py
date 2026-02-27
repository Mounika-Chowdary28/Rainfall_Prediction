import argparse
import requests
import pandas as pd
import numpy as np
import datetime
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

START_DATE = "2020-01-01"
END_DATE = "2024-12-31"
THRESHOLD_MM = 0.1


def try_ip_services():
    """Try a list of IP geolocation services; return (lat, lon, source_str) or None."""
    services = [
        ("https://ipinfo.io/json", lambda j: tuple(map(float, j["loc"].split(","))), "ipinfo"),
        ("https://ipwho.is/", lambda j: (float(j.get("latitude")), float(j.get("longitude"))), "ipwho.is"),
        ("http://www.geoplugin.net/json.gp", lambda j: (float(j.get("geoplugin_latitude")), float(j.get("geoplugin_longitude"))), "geoplugin"),
        ("http://ip-api.com/json", lambda j: (float(j.get("lat")), float(j.get("lon"))), "ip-api"),
    ]
    for (url, extractor, name) in services:
        try:
            r = requests.get(url, timeout=5)
            r.raise_for_status()
            j = r.json()
            coords = extractor(j)
            if coords and not any([c is None for c in coords]):
                return coords[0], coords[1], name
        except Exception:
            continue
    return None


def get_coords(args):
    
    if args.lat is not None and args.lon is not None:
        lat = float(args.lat); lon = float(args.lon)
        print("Detecting location (manual override)...")
        print(f"Using coordinates: lat={lat:.6f}, lon={lon:.6f}  (source: manual override)\n")
        return lat, lon

    if args.no_auto:
        return manual_coords_prompt()

 
    res = try_ip_services()
    if res:
        lat, lon, source = res
        
        print("Detecting location (approximate) via IP geolocation...")
        if source == "ipinfo":
            
            try:
                r = requests.get("https://ipinfo.io/json", timeout=5).json()
                city = r.get("city"); region = r.get("region")
                print(f"Using coordinates: lat={lat:.6f}, lon={lon:.6f}  (source: ipinfo (city={city}, region={region}))\n")
            except:
                print(f"Using coordinates: lat={lat:.6f}, lon={lon:.6f}  (source: ipinfo)\n")
        else:
            
            print(f"Using coordinates: lat={lat:.6f}, lon={lon:.6f}  (source: {source})\n")
        return lat, lon


    return manual_coords_prompt()


def manual_coords_prompt():
    print("Unable to auto-detect location or auto-detection disabled.")
    print("Please enter your coordinates manually.\n")
    while True:
        try:
            lat = float(input("Enter Latitude: ").strip())
            lon = float(input("Enter Longitude: ").strip())
            print("\nUsing manual coordinates:")
            print(f"lat={lat:.6f}, lon={lon:.6f}\n")
            return lat, lon
        except Exception:
            print("Invalid input. Try again.\n")



def fetch_archive(lat, lon):
    print("Downloading historical archive for these coordinates (this may take ~10-20s)...")
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}&start_date={START_DATE}&end_date={END_DATE}"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
        "relative_humidity_2m_mean,surface_pressure_mean,wind_speed_10m_max&timezone=auto"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    j = r.json()
    if 'daily' not in j or not isinstance(j['daily'], dict):
        raise RuntimeError("Unexpected archive JSON format.")
    df = pd.DataFrame(j["daily"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    print(f"Historical rows: {len(df)}\n")
    return df


def prepare(df):
    df = df.rename(columns={
        "temperature_2m_max": "temp_max",
        "temperature_2m_min": "temp_min",
        "precipitation_sum": "precip",
        "relative_humidity_2m_mean": "rh_mean",
        "surface_pressure_mean": "press_mean",
        "wind_speed_10m_max": "wind_max"
    })
    df["precip_lag1"] = df["precip"].shift(1)
    df["temp_max_lag1"] = df["temp_max"].shift(1)
    df["rh_mean_lag1"] = df["rh_mean"].shift(1)
    df["month"] = df.index.month
    df["dayofyear"] = df.index.dayofyear
    df["rain"] = (df["precip"] > THRESHOLD_MM).astype(int)
    return df.dropna()


def train_models(df):
    features = ["temp_max", "temp_min", "rh_mean", "press_mean", "wind_max",
                "precip_lag1", "temp_max_lag1", "rh_mean_lag1", "month", "dayofyear"]
    X = df[features]
    y_cls = df["rain"]
    y_reg = df["precip"]

    X_train, X_test, y_cls_train, y_cls_test = train_test_split(X, y_cls, test_size=0.2, shuffle=False)
    _, _, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, shuffle=False)

    scaler = StandardScaler().fit(X_train)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    clf.fit(scaler.transform(X_train), y_cls_train)

    reg = RandomForestRegressor(n_estimators=200, random_state=42)
    reg.fit(scaler.transform(X_train), y_reg_train)

    
    y_cls_pred = clf.predict(scaler.transform(X_test))
    cls_accuracy = accuracy_score(y_cls_test, y_cls_pred)
    cls_precision = precision_score(y_cls_test, y_cls_pred, zero_division=0)
    cls_recall = recall_score(y_cls_test, y_cls_pred, zero_division=0)
    cls_f1 = f1_score(y_cls_test, y_cls_pred, zero_division=0)

    
    y_reg_pred = reg.predict(scaler.transform(X_test))
    reg_mae = mean_absolute_error(y_reg_test, y_reg_pred)
    reg_rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
    reg_r2 = r2_score(y_reg_test, y_reg_pred)

    metrics = {
        'classification': {
            'accuracy': cls_accuracy,
            'precision': cls_precision,
            'recall': cls_recall,
            'f1_score': cls_f1
        },
        'regression': {
            'mae': reg_mae,
            'rmse': reg_rmse,
            'r2_score': reg_r2
        }
    }

    return clf, reg, scaler, features, metrics


def fetch_forecast(lat, lon, days=7):
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        "&daily=temperature_2m_max,temperature_2m_min,relative_humidity_2m_mean,"
        "surface_pressure_mean,wind_speed_10m_max,precipitation_sum&timezone=auto"
    )
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    j = r.json()
    if 'daily' not in j or not isinstance(j['daily'], dict):
        raise RuntimeError("Unexpected forecast JSON format.")
    df = pd.DataFrame(j["daily"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    return df.iloc[:days]


def build_features(i, forecast_df, hist_df, features):
    row = forecast_df.iloc[i]
    date = row.name.date()

    if i == 0:
        yesterday = date - datetime.timedelta(days=1)
        try:
            y = hist_df.loc[str(yesterday)]
            if isinstance(y, pd.DataFrame): y = y.iloc[-1]
        except:
            y = hist_df.iloc[-1]
    else:
        y = forecast_df.iloc[i-1]

    feat = {
        "temp_max": float(row.get("temperature_2m_max", np.nan)),
        "temp_min": float(row.get("temperature_2m_min", np.nan)),
        "rh_mean": float(row.get("relative_humidity_2m_mean", np.nan)),
        "press_mean": float(row.get("surface_pressure_mean", np.nan)),
        "wind_max": float(row.get("wind_speed_10m_max", np.nan)),
        "precip_lag1": float(y.get("precipitation_sum", y.get("precip", 0.0))),
        "temp_max_lag1": float(y.get("temperature_2m_max", y.get("temp_max", 0.0))),
        "rh_mean_lag1": float(y.get("relative_humidity_2m_mean", y.get("rh_mean", 0.0))),
        "month": int(date.month),
        "dayofyear": int(date.timetuple().tm_yday)
    }

    for k, v in feat.items():
        if pd.isna(v):
            feat[k] = float(hist_df[k].mean()) if k in hist_df.columns else 0.0
    return np.array([[feat[f] for f in features]]), feat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat", type=float, default=None, help="Latitude (override auto-detect)")
    parser.add_argument("--lon", type=float, default=None, help="Longitude (override auto-detect)")
    parser.add_argument("--no-auto", action="store_true", help="Disable auto-detection and prompt for coords")
    args = parser.parse_args()

    lat, lon = get_coords(args)

    hist_raw = fetch_archive(lat, lon)
    hist_prep = prepare(hist_raw)
    clf, reg, scaler, features, metrics = train_models(hist_prep)
    
 
    print("Model Performance Metrics:")
    print("\nClassification Model (Rain YES/NO):")
    print(f"  Accuracy:  {metrics['classification']['accuracy']:.4f} ({metrics['classification']['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['classification']['precision']:.4f}")
    print(f"  Recall:    {metrics['classification']['recall']:.4f}")
    print(f"  F1-Score:  {metrics['classification']['f1_score']:.4f}")
    print("\nRegression Model (Rainfall Amount):")
    print(f"  MAE (Mean Absolute Error): {metrics['regression']['mae']:.4f} mm")
    print(f"  RMSE (Root Mean Squared Error): {metrics['regression']['rmse']:.4f} mm")
    print(f"  R² Score: {metrics['regression']['r2_score']:.4f}")
    print("\n" + "="*56)
    print("7-DAY RAINFALL FORECAST")
    print("="*56)
    
    forecast_df = fetch_forecast(lat, lon, days=7)

    for i in range(len(forecast_df)):
        Xf, feat_dict = build_features(i, forecast_df, hist_raw, features)
        Xs = scaler.transform(Xf)

        pred_cls = int(clf.predict(Xs)[0])
        mm = float(reg.predict(Xs)[0])

        date = forecast_df.index[i].date()
        print("--------------------------------------------------------")
        print(f"{date} ({date.strftime('%a')})")
        print(f"  Rain: {'YES' if pred_cls == 1 else 'NO'}")
        print(f"  Predicted Rainfall: {mm:.3f} mm")
    print("--------------------------------------------------------")


if __name__ == "__main__":
    main()
