"""Full outer merge of complete six-route and Two-Tower train candidate pools."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--six',required=True); p.add_argument('--tower',required=True); p.add_argument('--processed',default='data/processed/retailrocket'); p.add_argument('--output',required=True); p.add_argument('--chunk-samples',type=int,default=3000); a=p.parse_args()
    samples=pd.read_parquet(ROOT/a.processed/'train_samples.parquet').sort_values(['target_ts','session'],kind='mergesort').reset_index(drop=True)
    original=samples.session.astype(np.int64).to_numpy(); target_type=samples.target_type.astype(str).to_numpy()
    six_path=ROOT/a.six; tower_path=ROOT/a.tower; output=ROOT/a.output; output.parent.mkdir(parents=True,exist_ok=True)
    writer=None; total=0; counts=[]
    try:
        for start in range(0,len(samples),a.chunk_samples):
            end=min(start+a.chunk_samples,len(samples))
            six=pq.read_table(six_path,filters=[('sample_id','>=',start),('sample_id','<',end)]).to_pandas()
            tower=pq.read_table(tower_path,filters=[('session','>=',start),('session','<',end)],columns=['session','aid','source_rank','source_score']).to_pandas().rename(columns={'session':'sample_id','source_rank':'source_rank_two_tower','source_score':'source_score_two_tower'})
            merged=six.merge(tower,on=['sample_id','aid'],how='outer',validate='one_to_one')
            tower_present=merged.source_rank_two_tower.notna()
            merged['rrf_score']=merged.rrf_score.fillna(0.0)+np.where(tower_present,1.0/(60.0+merged.source_rank_two_tower.fillna(0.0)),0.0)
            merged['source_count']=merged.source_count.fillna(0).astype(np.int16)+tower_present.astype(np.int16)
            merged['best_source_rank']=np.fmin(merged.best_source_rank.fillna(np.inf),merged.source_rank_two_tower.fillna(np.inf)).astype(np.int16)
            merged['label']=merged.label.fillna(0).astype(np.int8)
            idx=merged.sample_id.astype(np.int64).to_numpy(); merged['target_type']=[target_type[i] for i in idx]; merged['session']=original[idx]
            merged=merged.sort_values(['sample_id','aid'],kind='mergesort').reset_index(drop=True)
            sizes=merged.groupby('sample_id',sort=False).size(); counts.extend(sizes.tolist()); total+=len(merged)
            table=pa.Table.from_pandas(merged,preserve_index=False)
            if writer is None: writer=pq.ParquetWriter(output,table.schema,compression='zstd')
            writer.write_table(table)
            if start==0 or end==len(samples) or (start//a.chunk_samples)%20==0: print(f'merged={end}/{len(samples)} rows={total}',flush=True)
    finally:
        if writer is not None: writer.close()
    c=np.asarray(counts); report={'samples':len(samples),'rows':total,'average_candidates':float(c.mean()),'p50':float(np.quantile(c,.5)),'p90':float(np.quantile(c,.9)),'p99':float(np.quantile(c,.99)),'max':int(c.max()),'candidate_truncation':None,'rrf_used_for_truncation':False,'test_evaluated':False}
    output.with_suffix('.metrics.json').write_text(json.dumps(report,indent=2)); print(report)
if __name__=='__main__': main()
