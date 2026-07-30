import argparse
import subprocess
import sys
from pathlib import Path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Master Orchestrator for Mountain NER Pipeline")
    parser.add_argument("--skip-scrape", action="store_true", help="Skip downloading data from Wikidata")
    parser.add_argument("--skip-data-gen", action="store_true", help="Skip generating synthetic dataset")
    parser.add_argument("--skip-train-crf", action="store_true", help="Skip training the CRF model")
    parser.add_argument("--skip-train-bert", action="store_true", help="Skip fine-tuning the BERT model")
    parser.add_argument("--text", type=str, default="Last year I visited Nepal to see Mount Everest, but next time I want to conquer K2.", help="Text for final evaluation")
    return parser.parse_args()

def run_script(script_name: str, args: list = None) -> None:
    """Runs a python script as a subprocess to keep memory clean."""
    if args is None:
        args = []
        
    script_path = Path("scripts") / script_name
    if not script_path.exists():
        print(f"❌ Error: Script {script_path} not found!")
        sys.exit(1)

    cmd = [sys.executable, str(script_path)] + args
    print(f"\n{'='*60}")
    print(f"🚀 RUNNING: {script_name}")
    print(f"{'='*60}")
    
    try:
        # Запускаємо процес і виводимо його логи в реальному часі
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ FAILED: {script_name} crashed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print(f"\n🛑 STOPPED: User interrupted {script_name}")
        sys.exit(1)

def main():
    args = parse_args()
    
    print("🏔️ Starting Mountain NER Full Pipeline...")

    # 1. Data Collection
    if not args.skip_scrape:
        run_script("scrape_mountains.py")
    else:
        print("⏭️ Skipping Data Scraping...")

    # 2. Data Generation
    if not args.skip_data_gen:
        run_script("create_dataset.py", ["--samples", "5", "--negatives", "2000"])
    else:
        print("⏭️ Skipping Data Generation...")

    # 3. Model Training (Classic ML)
    if not args.skip_train_crf:
        run_script("train_crf.py")
    else:
        print("⏭️ Skipping CRF Training...")

    # 4. Model Training (Deep Learning)
    if not args.skip_train_bert:
        # Можеш винести epochs та batch-size в аргументи main.py за бажанням
        run_script("train_bert.py", ["--epochs", "3", "--batch-size", "8"])
    else:
        print("⏭️ Skipping BERT Training...")

    # 5. Final Comparison & LLM Judge
    print("\n" + "✨"*30)
    print("✨ PIPELINE COMPLETE! RUNNING FINAL INFERENCE ✨")
    print("✨"*30)
    
    run_script("run_full_pipeline.py")

if __name__ == "__main__":
    main()