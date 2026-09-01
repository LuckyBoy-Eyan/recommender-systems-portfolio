"""Generate direct in-sample six-route candidates for multitask-ranker training."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
import joblib, numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.recall.full_catalog import recall_from_frozen_indexes
from src.recall.item2vec_ann import Item2VecANN, Item2VecEmbeddings
from scripts.build_rolling_oof_candidates import compact_candidates

def main():
    p=argparse.ArgumentParser(); p.add_argument('--processed',default='data/processed/retailrocket'); p.add_argument('--output',default='outputs/direct_train_candidates_six_routes_full'); p.add_argument('--chunk-size',type=int,default=3000); p.add_argument('--item2vec-topk',type=int,default=250); a=p.parse_args()
    processed=ROOT/a.processed; output=ROOT/a.output; output.mkdir(parents=True,exist_ok=True)
    samples=pd.read_parquet(processed/'train_samples.parquet').sort_values(['target_ts','session'],kind='mergesort').reset_index(drop=True)
    samples['sample_id']=np.arange(len(samples),dtype=np.int64); original=dict(zip(samples.sample_id.astype(int),samples.session.astype(int)))
    scoring=samples.copy(); scoring['session']=scoring['sample_id']
    indexes=joblib.load(processed/'frozen_recall_indexes.joblib'); ann=Item2VecANN(Item2VecEmbeddings.load(processed/'frozen_item2vec_embeddings_v2.npz'))
    path=output/'six_route_candidates.parquet'; writer=None; rows=hits=0
    try:
        for start in range(0,len(scoring),a.chunk_size):
            chunk=scoring.iloc[start:start+a.chunk_size]
            raw=pd.concat([recall_from_frozen_indexes(chunk,indexes),ann.recall(chunk,a.item2vec_topk)],ignore_index=True)
            # max_negatives=0 means retain the complete deduplicated candidate pool.
            # RRF remains an optional feature only and never controls candidate inclusion.
            compact,local_hits=compact_candidates(raw,chunk,max_negatives=0,topk=250)
            compact=compact.rename(columns={'session':'sample_id'}); compact['session']=compact.sample_id.map(original).astype(np.int64)
            table=pa.Table.from_pandas(compact,preserve_index=False)
            if writer is None: writer=pq.ParquetWriter(path,table.schema,compression='zstd')
            writer.write_table(table); rows+=len(compact); hits+=local_hits
            if start==0 or start+a.chunk_size>=len(scoring) or (start//a.chunk_size)%20==0: print(f'scored={min(start+a.chunk_size,len(scoring))}/{len(scoring)}',flush=True)
    finally:
        if writer is not None: writer.close()
    report={'samples':len(scoring),'rows':rows,'candidate_recall_before_injection':hits/len(scoring),'candidate_truncation':None,'rrf_used_for_truncation':False,'direct_in_sample_retrieval':True,'test_evaluated':False}
    (output/'metrics.json').write_text(json.dumps(report,indent=2)); print(report)
if __name__=='__main__': main()
