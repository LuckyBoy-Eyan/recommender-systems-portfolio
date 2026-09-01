"""Evaluate frozen PLE fusion models on the complete independent validation pool."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
import numpy as np, pandas as pd, pyarrow.parquet as pq, torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.train_ple_fusion_models import StaticFusion, ContextMLP, DynamicGate, READ, raw_logits, context
from scripts.analyze_rankers_v3 import ranks_for, metrics

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--features',required=True); ap.add_argument('--models',required=True); ap.add_argument('--processed',default='data/processed/retailrocket'); ap.add_argument('--output',required=True); ap.add_argument('--batch-rows',type=int,default=200000); a=ap.parse_args()
    model_dir=Path(a.models); specs={'static':StaticFusion(),'context_mlp':ContextMLP(10),'dynamic_gate':DynamicGate(10)}; means={}; stds={}; best={}
    for name,m in specs.items():
        ck=torch.load(model_dir/f'{name}.pt',map_location='cpu',weights_only=False); m.load_state_dict(ck['state_dict']); m.eval(); means[name]=ck['mean']; stds[name]=ck['std']; best[name]=ck['best_epoch']
    ids=[]; aids=[]; labels=[]; actions=[]; scores={k:[] for k in specs}; gate_weights=[]
    with torch.no_grad():
        for b in pq.ParquetFile(a.features).iter_batches(batch_size=a.batch_rows,columns=READ):
            f=b.to_pandas(); z=torch.from_numpy(raw_logits(f)); c=torch.from_numpy(context(f,means['static'],stds['static']))
            ids.append(f.sample_id.to_numpy(np.int64)); aids.append(f.aid.to_numpy(np.int64)); labels.append(f.label.to_numpy(np.int8)); actions.append(f.target_type.astype(str).to_numpy())
            for name,m in specs.items(): scores[name].append(m(z,c).numpy().astype(np.float32))
            gate_weights.append(torch.softmax(specs['dynamic_gate'].gate(c),1).numpy())
    frame=pd.DataFrame({'sample_id':np.concatenate(ids),'aid':np.concatenate(aids),'label':np.concatenate(labels),'target_type':np.concatenate(actions)})
    validation=pd.read_parquet(Path(a.processed)/'validation_samples.parquet').sort_values(['target_ts','session'],kind='mergesort').reset_index(drop=True)
    action=validation.target_type.astype(str).to_numpy(); novel=np.array([int(r.target_aid) not in set(map(int,r.history_aids)) for r in validation.itertuples()])
    report={'best_epoch':best,'models':{},'test_evaluated':False}
    for name,parts in scores.items(): report['models'][name]=metrics(ranks_for(frame,np.concatenate(parts)),action,novel)
    all_gate_weights=np.concatenate(gate_weights)
    report['dynamic_gate_average_weights']=(all_gate_weights.sum(0,dtype=np.float64)/len(all_gate_weights)).tolist()
    raw=specs['static'].raw.detach(); report['static_positive_weights']=torch.nn.functional.softplus(raw).tolist(); report['static_bias']=float(specs['static'].bias.detach())
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(report,indent=2,ensure_ascii=False)); print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__': main()
