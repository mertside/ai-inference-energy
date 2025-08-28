import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("LlamaViaHF", Path(__file__).resolve().parents[1] / "app-llama" / "LlamaViaHF.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_resolve_model_name_case_insensitive():
    engine = module.LlamaInferenceEngine(model_name="LLAMA-7B", device="cpu")
    assert engine.model_name == "huggyllama/llama-7b"
