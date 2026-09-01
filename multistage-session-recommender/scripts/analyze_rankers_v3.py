"""Unified, vectorized analysis for frozen V3 rankers on the full candidate pool."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT=Path(__file__).resolve().parents[1]
ACTIONS=np.array(['clicks','carts','orders']); METRIC_W=np.array([.1,.3,.6])

def ranks_for(frame, score):
    sid=frame.sample_id.to_numpy(np.int64); aid=frame.aid.to_numpy(np.int64); label=frame.label.to_numpy(bool); n=int(sid.max())+1
    ps=np.full(n,np.nan); pa=np.full(n,np.iinfo(np.int64).max,dtype=np.int64); ps[sid[label]]=score[label]; pa[sid[label]]=aid[label]
    target=ps[sid]; better=(score>target)|((score==target)&(aid<pa[sid])); counts=np.bincount(sid,weights=better,minlength=n)
    ranks=counts+1.; ranks[np.isnan(ps)]=np.inf; return ranks

def metrics(ranks, action, novel, subset=None):
    use=np.ones(len(ranks),bool) if subset is None else subset; out={}
    for k in (20,50,100):
        hit=ranks<=k; rr=np.where(hit,1/ranks,0); nd=np.where(hit,1/np.log2(ranks+1),0)
        out[f'recall_at_{k}']=float(hit[use].mean()); out[f'mrr_at_{k}']=float(rr[use].mean()); out[f'ndcg_at_{k}']=float(nd[use].mean())
        per=[]
        for a in ACTIONS:
            mask=use&(action==a); value=float(hit[mask].mean()); out[f'{a}_recall_at_{k}']=value; per.append(value)
        out[f'macro_weighted_recall_at_{k}']=float(np.dot(METRIC_W,per))
        nm=use&novel; out[f'novel_recall_at_{k}']=float(hit[nm].mean())
    out['candidate_recall']=float(np.isfinite(ranks[use]).mean()); return out

def main():
    output=ROOT/'outputs/ranker_v3_analysis'; output.mkdir(parents=True,exist_ok=True)
    validation=pd.read_parquet(ROOT/'data/processed/retailrocket/validation_samples.parquet').sort_values(['target_ts','session'],kind='mergesort').reset_index(drop=True)
    action=validation.target_type.astype(str).to_numpy(); novel=np.array([int(r.target_aid) not in set(map(int,r.history_aids)) for r in validation.itertuples()])
    split=len(validation)//2; tune=np.arange(len(validation))<split; holdout=~tune
    reports={}; frames={}
    for name in ('mmoe','ple'):
        f=pd.read_parquet(output/f'{name}_scores.parquet'); frames[name]=f
        score=f.final_score.to_numpy(np.float32); ranks=ranks_for(f,score)
        p=f[['score_clicks','score_carts','score_orders']].to_numpy(np.float32)
        violation=(p[:,0]+1e-7<p[:,1])|(p[:,1]+1e-7<p[:,2])
        reports[name]={'fixed_fusion':metrics(ranks,action,novel),
                       'fixed_fusion_tune':metrics(ranks,action,novel,tune),
                       'fixed_fusion_holdout':metrics(ranks,action,novel,holdout),
                       'ordinal_violation_candidate_rate':float(violation.mean()),'ordinal_violation_session_rate':float(pd.Series(violation).groupby(f.sample_id,sort=False).any().mean())}
    base=pd.read_parquet(ROOT/'outputs/ranker_datasets_v2/features/validation.parquet',columns=['sample_id','aid','label','rrf_score'])
    reports['rrf']={'fixed_fusion':metrics(ranks_for(base,base.rrf_score.to_numpy(np.float32)),action,novel)}
    grid=[]
    for wc in (.05,.1,.15,.2):
        for wa in (.1,.2,.3):
            for wo in (.2,.3,.4): grid.append((wc,wa,wo))
    for name,f in frames.items():
        probs=f[['score_clicks','score_carts','score_orders']].to_numpy(np.float32); best=None
        for weights in grid:
            ranks=ranks_for(f,probs@np.asarray(weights,np.float32)); score=metrics(ranks,action,novel,tune)['macro_weighted_recall_at_20']
            if best is None or score>best[0]: best=(score,weights,ranks)
        reports[name]['searched_fusion']={'weights':best[1],'tune':metrics(best[2],action,novel,tune),'holdout':metrics(best[2],action,novel,holdout),'full':metrics(best[2],action,novel)}
    reports['protocol']={'fusion_tune_samples':int(tune.sum()),'fusion_holdout_samples':int(holdout.sum()),'test_evaluated':False}
    (output/'analysis.json').write_text(json.dumps(reports,indent=2,ensure_ascii=False)); print(json.dumps(reports,ensure_ascii=False))
if __name__=='__main__': main()
