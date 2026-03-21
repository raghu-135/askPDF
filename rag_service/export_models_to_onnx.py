from optimum.onnxruntime import (
    ORTModelForFeatureExtraction,
    ORTModelForSequenceClassification
)
from transformers import AutoTokenizer
from pathlib import Path

models = {
    "bge_m3": "BAAI/bge-m3",
    "bge_reranker": "BAAI/bge-reranker-v2-m3"
}

output_dir = Path("/models/onnx")
output_dir.mkdir(parents=True, exist_ok=True)

#
# Export BGE-M3 (embedding model)
#
print("Exporting BGE-M3 to ONNX...")
bge_m3_path = output_dir / "bge-m3"
bge_m3_path.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(models["bge_m3"], trust_remote_code=True)
model = ORTModelForFeatureExtraction.from_pretrained(
    models["bge_m3"],
    export=True,
    trust_remote_code=True
)
model.save_pretrained(bge_m3_path)
tokenizer.save_pretrained(bge_m3_path)

#
# Export BGE reranker
#
print("Exporting BGE reranker to ONNX...")
reranker_path = output_dir / "bge-reranker-v2-m3"
reranker_path.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(models["bge_reranker"], trust_remote_code=True)
model = ORTModelForSequenceClassification.from_pretrained(
    models["bge_reranker"],
    export=True,
    trust_remote_code=True
)
model.save_pretrained(reranker_path)
tokenizer.save_pretrained(reranker_path)

print("Export complete.")
