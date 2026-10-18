"""Hardware sizing only: random disposable weights, backward, NO optimizer."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from tac.neural.model import ModelConfig, TACTransformer, count_parameters

torch.set_num_threads(4)
cfg = ModelConfig(vocab_size=8192, context=1024, width=512, heads=8,
                  layers=8, ff_width=2048, tie_embeddings=True, rope=True)
torch.manual_seed(4317)
torch.cuda.set_per_process_memory_fraction(.73)
model = TACTransformer(cfg).cuda().train()
results = []
for batch, length in [(2, 512), (4, 512), (2, 1024)]:
    torch.cuda.reset_peak_memory_stats()
    x = torch.randint(3, cfg.vocab_size, (batch, length), device='cuda')
    started = None
    for iteration in range(13):
        model.zero_grad(set_to_none=True)
        if iteration == 3:
            torch.cuda.synchronize()
            started = time.monotonic()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, loss = model(x, x)
        loss.backward()
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    results.append(dict(batch=batch, length=length, measured_backward_batches=10,
                        seconds=elapsed, tokens_per_second=batch * length * 10 / elapsed,
                        peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                        peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20))
    del x, loss
    torch.cuda.empty_cache()
out = dict(parameters=count_parameters(cfg), device=torch.cuda.get_device_name(),
           optimizer_steps=0, active_checkpoint_loaded=False, dtype='bfloat16', measurements=results)
folder = ROOT / 'reports' / 'tac_v3' / 'rodada_20261018_01'
folder.mkdir(parents=True, exist_ok=True)
(folder / 'capacidade_gpu.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
print(json.dumps(out, indent=2))
