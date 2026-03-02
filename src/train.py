from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler    
from preprocess import load_process
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA = BASE_DIR / "data" / "raw" / "flood_dataset_classification.csv"
MODEL_PATH = BASE_DIR / "models" / "model.pkl"

def train():

    data = load_process(RAW_DATA)
    
    X = data.drop(['occured'], axis=1)
    y = data['occured']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    xgb = XGBClassifier(
    scale_pos_weight=944/304,
    n_estimators=500,
    colsample_bytree=0.6,
    max_depth=10,
    learning_rate=0.1,
)

    xgb.fit(X_train_s, y_train)
    
    y_pred = xgb.predict(X_test_s)
    print(classification_report(y_pred, y_test))
    print(xgb.score(X_test_s, y_test))
    
  
if __name__ == "__main__":
    train() 
    