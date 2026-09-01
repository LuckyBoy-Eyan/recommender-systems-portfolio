"""Build the untouched full test candidate pool and PLE features exactly once."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
import joblib, numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.recall.full_catalog import recall_from_frozen_indexes
from src.recall.item2vec_ann import Item2VecANN,Item2VecEmbeddings
from scripts.build_rolling_oof_candidates import compact_candidates
from scripts.build_ranker_features import prepare_samples,snapshot_item_features,enrich_file

def main():
 p=argparse.ArgumentParser();p.add_argument('--processed',default='data/processed/retailrocket');p.add_argument('--tower',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 processed=ROOT/a.processed;out=ROOT/a.output;out.mkdir(parents=True,exist_ok=True)
 test=pd.read_parquet(processed/'test_samples.parquet').sort_values(['target_ts','session'],kind='mergesort').reset_index(drop=True);test['sample_id']=np.arange(len(test),dtype=np.int64)
 indexes=joblib.load(processed/'frozen_recall_indexes.joblib');ann=Item2VecANN(Item2VecEmbeddings.load(processed/'frozen_item2vec_embeddings_v2.npz'))
 six=pd.concat([recall_from_frozen_indexes(test,indexes),ann.recall(test,250)],ignore_index=True);tower=pd.read_parquet(ROOT/a.tower);raw=pd.concat([six,tower],ignore_index=True)
 compact,hits=compact_candidates(raw,test,max_negatives=0,topk=300,inject_missing_positives=False,prioritize_positives=False)
 mapping=dict(zip(test.session.astype(int),test.sample_id.astype(int)));original=dict(zip(test.sample_id.astype(int),test.session.astype(int)))
 compact['sample_id']=compact.session.map(mapping).astype(np.int64);compact['session']=compact.sample_id.map(original).astype(np.int64);compact['target_ts']=compact.sample_id.map(dict(zip(test.sample_id,test.target_ts))).astype(np.int64)
 candidates=out/'test_candidates.parquet';compact.to_parquet(candidates,index=False)
 session,pairs=prepare_samples(processed/'test_samples.parquet');cutoff=int(test.target_ts.min())
 item=snapshot_item_features(cutoff,pd.read_parquet(processed/'item_category_changes.parquet'),pd.read_parquet(processed/'item_availability_changes.parquet'),pd.read_parquet(processed/'category_paths.parquet'),pd.read_parquet(processed/'item_first_seen.parquet'))
 report=enrich_file(candidates,out/'test_features.parquet',session,pairs,item,training=False,batch_size=250000)
 metrics={'samples':len(test),'rows':len(compact),'candidate_hit_rate':hits/len(test),'average_candidates':len(compact)/len(test),'feature_report':report,'candidate_truncation':None,'rrf_used_for_truncation':False,'test_evaluated':True}
 (out/'candidate_metrics.json').write_text(json.dumps(metrics,indent=2));print(metrics)
if __name__=='__main__':main()
