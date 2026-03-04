from pathlib import Path
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler    
from preprocess import load_process
from sklearn.metrics import classification_report
import joblib
from sklearn.model_selection import GridSearchCV

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA = BASE_DIR / "data" / "raw" / "flood_dataset_classification.csv"
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

def train():

    data = load_process(RAW_DATA)
    print(data.info())
    
    X = data.drop(['occured'], axis=1)
    y = data['occured']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    xgb_model = XGBClassifier(
    scale_pos_weight=944/304,
    n_estimators=500,
    colsample_bytree=0.6,
    max_depth=10,
    learning_rate=0.1,
)
    # training the model
    xgb_model.fit(X_train_s, y_train)
    
    y_pred = xgb_model.predict(X_test_s)
    
    # to see if the model has high f1-score
    print(classification_report(y_pred, y_test))
    print(xgb_model.score(X_test_s, y_test))
    
  
    params = {
        "n_estimators" : [200, 500],
        "colsample_bytree" : [0.2, 0.6, 1],
        "max_depth" : [8, 10],
        "learning_rate" : [0.05, 0.1]
    }
    
    # hyperparameters tuning to create the best model
    grid_search = GridSearchCV(xgb_model, params, cv=3, return_train_score=True, verbose=2, n_jobs=-1, scoring="f1_macro")
    grid_search.fit(X_train_s, y_train)
    best_estimators = grid_search.best_estimator_
    
    print(grid_search.best_estimator_)
    print(best_estimators.score(X_test_s, y_test))
    
    #saving the model
    
    #joblib.dump(scaler, SCALER_PATH)
    #joblib.dump(xgb_model, MODEL_PATH)
  
if __name__ == "__main__":
    train() 
    