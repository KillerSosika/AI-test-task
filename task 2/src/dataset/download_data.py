import os
import shutil
from pathlib import Path
import kagglehub
from dotenv import load_dotenv

def setup_dataset():
    load_dotenv()
    
    if not os.getenv("KAGGLE_USERNAME") or not os.getenv("KAGGLE_KEY"):
        raise ValueError("❌ KAGGLE_USERNAME or KAGGLE_KEY was not found in the .env file!")

    project_root = Path(__file__).resolve().parents[2]
    raw_data_dir = project_root / "data" / "raw"

    print("⏳ Downloading the dataset via Kaggle API...")
    cache_path = Path(kagglehub.dataset_download("salmaadell/eurosat-rgb"))
    print(f"✅ Dataset downloaded to the system cache: {cache_path}")

    source_dir = cache_path
    
    subdirs = [d for d in source_dir.iterdir() if d.is_dir()]
    if len(subdirs) == 1 and subdirs[0].name == "EuroSAT_RGB":
        source_dir = subdirs[0]
        print(f"🛠 Found nested folder '{source_dir.name}'. Aligning the structure...")

    raw_data_dir.mkdir(parents=True, exist_ok=True)
        
    print(f"🚚 Copying classes directly into {raw_data_dir}...")
    
    for item in source_dir.iterdir():
        target_item = raw_data_dir / item.name
        
        if target_item.exists():
            if target_item.is_dir():
                shutil.rmtree(target_item)
            else:
                target_item.unlink()
        
        if item.is_dir():
            shutil.copytree(item, target_item)
        else:
            shutil.copy(item, target_item)
    
    print("🎉 Done! The data is stored in the expected location:")
    print(f"📂 {raw_data_dir.absolute()}")

if __name__ == "__main__":
    setup_dataset()