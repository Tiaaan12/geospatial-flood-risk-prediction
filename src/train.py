from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from preprocess import load_process

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA = BASE_DIR / "data" / "raw" / "flood_dataset_classification.csv"
MODEL_PATH = BASE_DIR / "models" / "model.pkl"

def train():
    
    data = load_process(RAW_DATA)
    
    print(data)

if __name__ == "__main__":
    train() 
    