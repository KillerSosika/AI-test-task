import sys
from pathlib import Path
import transformers

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.models.bert_model import BERTModel
from src.llm.llm_judge import LocalLLMJudge

def main():
    # Disable extra Hugging Face warnings to keep the output clean
    transformers.logging.set_verbosity_error()
    
    config = load_config()
    text = "Last year I visited Nepal to see Mount Everest, but next time I want to conquer K2 and maybe a small hill near my house."
    
    print("=" * 60)
    print(" 🚀 FULL NER PIPELINE (BERT + LLM JUDGE)")
    print("=" * 60)
    print(f"Input Text: {text}\n")
    
    # ---------------------------------------------------------
    # Phase 1: BERT Extraction (High Recall)
    # ---------------------------------------------------------
    print("--- Phase 1: BERT Extraction ---")
    bert_dir = config['paths']['models_finetuned']
    
    try:
        bert_predictor = BERTModel(model_path=bert_dir)
    except Exception as e:
        print(f"❌ Error loading BERT from '{bert_dir}': {e}")
        print("Did you run 'python scripts/train_bert.py' first?")
        sys.exit(1)

    bert_preds = bert_predictor.predict(text)
    
    candidates = []
    for entity in bert_preds:
        word = entity.get('word', '')
        score = entity.get('score', 0.0)
        candidates.append(word)
        print(f"  🔍 Found candidate: {word:<15} (confidence: {score:.4f})")

    if not candidates:
        print("\n✅ FINAL VERIFIED MOUNTAINS:")
        print("  (No candidates found by BERT in Phase 1)")
        return

    # ---------------------------------------------------------
    # Phase 2: LLM Validation (High Precision)
    # ---------------------------------------------------------
    print("\n--- Phase 2: LLM Validation (Ollama) ---")
    print(f"Sending candidates to Judge: {candidates}...")
    
    # Initialize our local judge
    judge = LocalLLMJudge(model_name="llama3")
    final_mountains = judge.filter_entities(candidates)
    
    # ---------------------------------------------------------
    # Results Output
    # ---------------------------------------------------------
    print("\n✅ FINAL VERIFIED MOUNTAINS:")
    
    if not final_mountains:
        print("  (LLM returned an empty list. All candidates were filtered out or parsing failed.)")
    else:
        for m in final_mountains:
            print(f"  🏔️  {m}")
            
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()