"""Train static, contextual MLP and dynamic-gate fusion models on frozen PLE outputs."""
from __future__ import annotations
import argparse, copy, json
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq, torch
from torch import nn
from torch.nn import functional as F

LOGITS=['score_clicks','score_carts','score_orders']
CONT=['history_length','history_unique_items','session_span_minutes','source_score_hybrid_popular','source_score_category','candidate_age_days','candidate_in_history']
READ=['sample_id','aid','target_type','label','last_type_id']+LOGITS+CONT

class StaticFusion(nn.Module):
    def __init__(self): super().__init__(); self.raw=nn.Parameter(torch.zeros(3)); self.bias=nn.Parameter(torch.zeros(()))
    def forward(self,z,c): return (z*F.softplus(self.raw)).sum(1)+self.bias
class ContextMLP(nn.Module):
    def __init__(self,d): super().__init__(); self.net=nn.Sequential(nn.Linear(d+3,8),nn.ReLU(),nn.Dropout(.2),nn.Linear(8,1))
    def forward(self,z,c): return self.net(torch.cat([z,c],1)).squeeze(1)
class DynamicGate(nn.Module):
    def __init__(self,d): super().__init__(); self.gate=nn.Sequential(nn.Linear(d,8),nn.ReLU(),nn.Linear(8,3)); self.bias=nn.Parameter(torch.zeros(()))
    def forward(self,z,c): return (z*torch.softmax(self.gate(c),1)).sum(1)+self.bias

def raw_logits(frame):
    p=np.clip(frame[LOGITS].to_numpy(np.float32),1e-6,1-1e-6); return np.log(p/(1-p)).astype(np.float32)
def context(frame,mean,std):
    x=(frame[CONT].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(np.float32)-mean)/std
    t=frame.last_type_id.fillna(-1).astype(int).to_numpy(); one=np.stack([t==0,t==1,t==2],1).astype(np.float32)
    return np.concatenate([x,one],1)
def keep_rows(frame,seed,p):
    pos=frame.label.to_numpy(bool)
    with np.errstate(over='ignore'):
        mixed=frame.sample_id.to_numpy(np.uint64)*np.uint64(11400714819323198485)+frame.aid.to_numpy(np.uint64)*np.uint64(14029467366897019727)+np.uint64(seed)*np.uint64(1609587929392839161)
    u=(mixed>>np.uint64(11)).astype(np.float64)/float(1<<53); return pos|(u<p)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--train',required=True); ap.add_argument('--validation',required=True); ap.add_argument('--output',required=True); ap.add_argument('--epochs',type=int,default=10); ap.add_argument('--batch-rows',type=int,default=200000); ap.add_argument('--batch-size',type=int,default=4096); ap.add_argument('--device',default='auto'); a=ap.parse_args()
    output=Path(a.output); output.mkdir(parents=True,exist_ok=True); train=Path(a.train); split=406000; keep=.205
    total=np.zeros(len(CONT)); squares=np.zeros(len(CONT)); count=0
    for b in pq.ParquetFile(train).iter_batches(batch_size=a.batch_rows,columns=['sample_id']+CONT):
        f=b.to_pandas(); f=f[f.sample_id<split]; x=f[CONT].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(np.float64); total+=x.sum(0); squares+=(x*x).sum(0); count+=len(x)
    mean=(total/count).astype(np.float32); std=np.sqrt(np.maximum(squares/count-(total/count)**2,1e-8)).astype(np.float32)
    device='mps' if a.device=='auto' and torch.backends.mps.is_available() else ('cuda' if a.device=='auto' and torch.cuda.is_available() else ('cpu' if a.device=='auto' else a.device))
    models={'static':StaticFusion(),'context_mlp':ContextMLP(10),'dynamic_gate':DynamicGate(10)}; models={k:v.to(device) for k,v in models.items()}; opts={k:torch.optim.AdamW(v.parameters(),lr=1e-3,weight_decay=1e-5) for k,v in models.items()}
    best={k:(float('inf'),None,0) for k in models}; history=[]; pos_weight=torch.tensor(12.65,device=device)
    for epoch in range(1,a.epochs+1):
        losses={k:[] for k in models}
        for b in pq.ParquetFile(train).iter_batches(batch_size=a.batch_rows,columns=READ):
            f=b.to_pandas(); f=f[(f.sample_id<split)&keep_rows(f,2026+epoch,keep)];
            if f.empty: continue
            order=np.random.default_rng(2026+epoch).permutation(len(f))
            zall=raw_logits(f); call=context(f,mean,std); yall=f.label.to_numpy(np.float32)
            for start in range(0,len(f),a.batch_size):
                ix=order[start:start+a.batch_size]; z=torch.from_numpy(zall[ix]).to(device); c=torch.from_numpy(call[ix]).to(device); y=torch.from_numpy(yall[ix]).to(device)
                for name,m in models.items():
                    m.train(); loss=F.binary_cross_entropy_with_logits(m(z,c),y,pos_weight=pos_weight); opts[name].zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),5); opts[name].step(); losses[name].append(float(loss.detach().cpu()))
        val_loss={k:[] for k in models}
        with torch.no_grad():
            for b in pq.ParquetFile(train).iter_batches(batch_size=a.batch_rows,columns=READ):
                f=b.to_pandas(); f=f[(f.sample_id>=split)&keep_rows(f,9999,keep)];
                if f.empty: continue
                z=torch.from_numpy(raw_logits(f)).to(device); c=torch.from_numpy(context(f,mean,std)).to(device); y=torch.from_numpy(f.label.to_numpy(np.float32)).to(device)
                for name,m in models.items(): m.eval(); val_loss[name].append(float(F.binary_cross_entropy_with_logits(m(z,c),y,pos_weight=pos_weight)))
        row={'epoch':epoch,'train_loss':{k:float(np.mean(v)) for k,v in losses.items()},'holdout_loss':{k:float(np.mean(v)) for k,v in val_loss.items()}}; history.append(row); print(row,flush=True)
        for k,m in models.items():
            if row['holdout_loss'][k]<best[k][0]: best[k]=(row['holdout_loss'][k],copy.deepcopy(m.state_dict()),epoch)
    for k,m in models.items(): m.load_state_dict(best[k][1]); torch.save({'state_dict':m.cpu().state_dict(),'mean':mean,'std':std,'best_epoch':best[k][2]},output/f'{k}.pt')
    report={'history':history,'best_epoch':{k:v[2] for k,v in best.items()},'device':device,'test_evaluated':False}; (output/'metrics.json').write_text(json.dumps(report,indent=2)); print(report)
if __name__=='__main__': main()
