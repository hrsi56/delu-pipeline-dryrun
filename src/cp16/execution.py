"""Ordered pre-run admission, fixed replay and scoring; no recipe search."""
from __future__ import annotations
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from cp15.data import LABELS, sha, array_hash, origin_utc, prepare
from .budget import Budget, atomic, counted_fits
from .inputs import load, identities, expected, SavedComponents
from .residuals import SharedResidualState


def check_protocol(root):
    path=root/'reports/v2-causal/protocol.json'
    p=json.loads(path.read_text())
    for name, digest in p['implementation_sha256'].items():
        if sha(root/name)!=digest: raise ValueError('implementation changed since pre-run freeze: '+name)
    # The complete pre-run protocol must already be committed before fitting.
    committed=subprocess.check_output(['git','show','HEAD:reports/v2-causal/protocol.json'],cwd=root)
    if committed!=path.read_bytes():raise ValueError('pre-run protocol is not committed')
    identities(root)
    return p


def compare_fit(cache, data, day, budget, label):
    budget.reserve(policy_days=2)  # Real-data representative forecast reproduction.
    saved=cache.get(day)
    if saved is None:raise ValueError('representative cache day unavailable')
    rows, central, logs=saved
    results={}
    with counted_fits(budget) as fit:
        for policy in ('A1','B2'):
            pred, observed=fit(data,day,policy,rows=rows)
            np.testing.assert_allclose(pred,central[policy],atol=1e-8,rtol=1e-10)
            original={int(x['local_hour']):x for x in logs[policy]}
            for row in observed:
                for field in ('model_sha256','imputer_sha256','scaler_sha256','train_target_sha256',
                              'normalization_rows_sha256','hour_train_rows_sha256','inner_train_rows_sha256','validation_rows_sha256'):
                    if row[field]!=original[row['local_hour']][field]:raise ValueError('fit reproduction fingerprint mismatch: '+field)
            results[policy]={'max_abs_difference':float(np.max(np.abs(pred-central[policy]))),
                             'fingerprints_match':True,'primitive_calls':sum(x['solver_calls'] for x in observed)}
    return {'label':label,'fold':cache.fold,'day':str(day),'results':results}


def component_digest(item):
    return hashlib.sha256(json.dumps({k:v for k,v in item.items() if k!='content_sha256'},sort_keys=True,allow_nan=False,default=str).encode()).hexdigest()


def save_vectors(path, frames):
    if frames:
        temp=path.with_name(path.name+'.tmp')
        pd.concat(frames,ignore_index=True).to_parquet(temp,index=False)
        temp.replace(path)


def component(data,cache,day,lineage,budget):
    rows=data.rows(day)
    if not len(rows):return rows, {'A1':np.array([]),'B2':np.array([])}, 'original_no_eligible_hours'
    saved=cache.get(day)
    if saved is not None:return saved[0],saved[1],'verified_cp15_cache'
    key=cache.fold+':'+str(day)
    if key in lineage['new_components']:
        item=lineage['new_components'][key]
        if item.get('content_sha256')!=component_digest(item):raise ValueError('new-component content identity mismatch')
        if item.get('protocol_sha256')!=lineage['protocol_sha256'] or item.get('day')!=str(day) or item.get('fold')!=cache.fold or item.get('origin_utc')!=str(origin_utc(day).tz_convert('UTC')):raise ValueError('new-component origin/protocol identity mismatch')
        if item['input_fingerprint']!=lineage['input_fingerprint'] or item['timestamp_utc']!=list(map(str,data.index[rows])):
            raise ValueError('stale/wrong new-component identity')
        if item['scale_sha256']!=array_hash(data.scale[rows]):raise ValueError('new-component scale identity mismatch')
        centers={p:np.asarray(item['central'][p],float) for p in ('A1','B2')}
        if any(v.shape!=(len(rows),) or not np.isfinite(v).all() for v in centers.values()):raise ValueError('invalid cached central vector')
        return rows,centers,'verified_cp16_cache'
    centers={};logs={}
    with counted_fits(budget,main=True) as fit:
        for policy in ('A1','B2'):centers[policy],logs[policy]=fit(data,day,policy,rows=rows)
    lineage['new_components'][key]={'day':str(day),'fold':cache.fold,'origin_utc':str(origin_utc(day).tz_convert('UTC')),
        'timestamp_utc':list(map(str,data.index[rows])),'input_fingerprint':lineage['input_fingerprint'],
        'protocol_sha256':lineage['protocol_sha256'],'scale_sha256':array_hash(data.scale[rows]),'central':{p:c.tolist() for p,c in centers.items()},'fits':logs}
    lineage['new_components'][key]['content_sha256']=component_digest(lineage['new_components'][key])
    return rows,centers,'fresh_components'


def frame(data,fold,day,rows,centers,predictions):
    frames=[]
    for policy,q in predictions.items():
        item=pd.DataFrame({'fold':fold,'policy':policy,'timestamp_utc':data.index[rows],
            'delivery_date':day,'origin_utc':origin_utc(day).tz_convert('UTC'),'y_true':data.y[rows],
            'central':centers['A1']/2+centers['B2']/2,'scale':data.scale[rows],
            'level':data.level[rows],'evidence_class':'development_post_selection'})
        for j,name in enumerate(LABELS):item[name]=q[:,j]
        frames.append(item)
    return frames


def admission(root,out,budget):
    manifest=json.loads((out/'input-manifest.json').read_text())
    lineage={'schema':'cp16-lineage-v1','input_fingerprint':hashlib.sha256(json.dumps(manifest['input_sha256'],sort_keys=True).encode()).hexdigest(),
        'protocol_sha256':sha(root/'reports/v2-causal/protocol.json'),'new_components':{},'cache_reproduction':[],'admission':[],'origins':[], 'states':{},
        'execution_stage':'training_only_admission_in_progress'}
    if (out/'lineage.json').exists():raise ValueError('existing admission evidence; no automatic retry/overwrite')
    for f in manifest['folds']:
        first=date.fromisoformat(f['evaluation_start'])
        data,_=load(root,before=first);cache=SavedComponents(root,f['fold'],data)
        first_cached=min(date.fromisoformat(x['day']) for x in f['origins'] if x['cache_present'])
        lineage['cache_reproduction'].append(compare_fit(cache,data,first_cached,budget,'training-only cache admission'))
        state=SharedResidualState();truth=pd.Series(data.y,index=data.index)
        admission_days={x['day'] for x in f['admission']}
        for d in pd.date_range(f['warmup_start'],first-timedelta(days=1)).date:
            budget.reserve(policy_days=2)
            state.release(d,lambda ix:truth.reindex(ix).to_numpy())
            rows,centers,source=component(data,cache,d,lineage,budget)
            metadata={}
            if str(d) in admission_days:
                q,metadata=state.predict(d,data.index[rows],centers['A1'],centers['B2'],data.scale[rows])
                lineage['admission'].append({'fold':f['fold'],'day':str(d),'policy_days':2,
                    'n_hours':len(rows),'finite_ordered':True,'vector_sha256':{p:array_hash(v) for p,v in q.items()},**metadata})
            if len(rows):state.issue(d,data.index[rows],centers['A1'],centers['B2'],data.scale[rows])
            lineage['origins'].append({'fold':f['fold'],'day':str(d),'phase':'training_only',
                'source':source,'n_hours':len(rows),**metadata})
            lineage['states'][f['fold']]=state.to_dict()
            atomic(out/'lineage.json',lineage)
            print(f['fold'],d,source,'admission',str(d) in admission_days,flush=True)
        lineage['states'][f['fold']]=state.to_dict()
        atomic(out/'lineage.json',lineage)
    if len(lineage['admission'])!=35:raise ValueError('missing admission date')
    lineage['execution_stage']='training_only_admission_complete_frozen_before_outer_scoring'
    atomic(out/'lineage.json',lineage)


def comparison(root,out,budget):
    lineage=json.loads((out/'lineage.json').read_text())
    if lineage['execution_stage']!='training_only_admission_complete_frozen_before_outer_scoring':raise ValueError('admission not complete or comparison already run')
    data,_=load(root);truth=pd.Series(data.y,index=data.index);frames=[];failures=[]
    for f in data.spec.development_folds:
        state=SharedResidualState.from_dict(lineage['states'][f.name]);cache=SavedComponents(root,f.name,data)
        for d in pd.date_range(f.evaluation.start,f.evaluation.end).date:
            budget.reserve(policy_days=2)
            try:
                state.release(d,lambda ix:truth.reindex(ix).to_numpy())
                rows,centers,source=component(data,cache,d,lineage,budget)
                if not len(rows):
                    lineage['origins'].append({'fold':f.name,'day':str(d),'phase':'evaluation','source':source,'n_hours':0})
                    continue
                q,metadata=state.predict(d,data.index[rows],centers['A1'],centers['B2'],data.scale[rows])
                frames.extend(frame(data,f.name,d,rows,centers,q))
                state.issue(d,data.index[rows],centers['A1'],centers['B2'],data.scale[rows])
                lineage['origins'].append({'fold':f.name,'day':str(d),'phase':'evaluation','source':source,'n_hours':len(rows),**metadata})
                save_vectors(out/'predictions.parquet',frames)
                lineage['states'][f.name]=state.to_dict()
                lineage['partial_vectors_sha256']=sha(out/'predictions.parquet')
                atomic(out/'lineage.json',lineage)
            except Exception as exc:
                failures.append({'fold':f.name,'day':str(d),'cause':repr(exc),'expected_hours':len(data.rows(d))})
                pd.DataFrame(failures).to_csv(out/'failures.csv',index=False)
                atomic(out/'lineage.json',lineage)
                if frames:pd.concat(frames,ignore_index=True).to_parquet(out/'predictions.parquet',index=False)
                raise
        lineage['states'][f.name]=state.to_dict();atomic(out/'lineage.json',lineage)
    saved=pd.read_parquet(root/'reports/cp15/predictions.parquet',filters=[('policy','in',['B0','B1','B2','B3','A1'])])
    saved['evidence_class']='development_post_selection'
    pred=pd.concat([saved,*frames],ignore_index=True)
    if len(pred)!=75229:raise ValueError('missing original eligible predictions')
    pred.to_parquet(out/'predictions.parquet',index=False)
    pd.DataFrame(columns=['fold','day','cause','expected_hours']).to_csv(out/'failures.csv',index=False)
    lineage['execution_stage']='comparison_vectors_complete_not_scored'
    atomic(out/'lineage.json',lineage)


def controls(root,out,budget):
    # Component forecasts only; no outer scoring or extra residual recipe.
    data,_=load(root);day=date(2020,7,1);cache=SavedComponents(root,'fold_1',data)
    result=[compare_fit(cache,data,day,budget,'origin/causal base')]
    rows,original,_=cache.get(day)
    masked=data.frame.copy();masked.loc[masked.delivery_date>=day,'price_eur_mwh']=np.nan
    masked.loc[masked.delivery_date>day,'load_forecast_mw']=1e8
    changed=data.frame.copy();changed.loc[changed.delivery_date.eq(day-timedelta(days=1)),'price_eur_mwh']+=500
    with counted_fits(budget) as fit:
        for name,raw in [('delivery_day_mask',masked),('available_d1_mutation',changed)]:
            budget.reserve(policy_days=2)
            altered=prepare(raw,data.p,data.spec);record={'control':name,'policies':{}}
            for policy in ('A1','B2'):
                pred,logs=fit(altered,day,policy,rows=rows)
                delta=float(np.max(np.abs(pred-original[policy])))
                if name=='delivery_day_mask' and delta!=0.:raise ValueError('delivery-day leakage')
                if name=='available_d1_mutation' and delta<=0.:raise ValueError('positive causal control insensitive')
                record['policies'][policy]={'max_abs_difference':delta,'fit_records':len(logs)}
            result.append(record)
    atomic(root/'docs/track-b/evidence/cp-16/causal-controls.json',result)


def score(root,out,budget):
    budget.reserve(reference_passes=1,analysis_passes=1)
    from .scoring import evaluate
    pred=pd.read_parquet(out/'predictions.parquet')
    tables=evaluate(pred,expected(root))
    for name,value in tables.items():
        if isinstance(value,pd.DataFrame):value.to_csv(out/f'{name}.csv',index=False)
        else:
            lineage=json.loads((out/'lineage.json').read_text())
            lineage['research_'+name]=value
            lineage['execution_stage']='scored_development_post_selection'
            atomic(out/'lineage.json',lineage)


def execute(root,args):
    p=check_protocol(root)
    if p.get('execution_status','').startswith('BLOCKED'):
        raise RuntimeError('CP-16 scientific execution is blocked pending amended authority and budget')
    out=args.output;out.mkdir(parents=True,exist_ok=True);budget=Budget(args.ledger)
    try:
        if args.job=='admission':admission(root,out,budget)
        elif args.job=='comparison':comparison(root,out,budget)
        elif args.job=='score':score(root,out,budget)
        elif args.job=='controls':controls(root,out,budget)
        else:raise ValueError('verification is performed by the independent checker')
    except Exception as exc:
        budget.event('failed_job',job=args.job,cause=repr(exc))
        # Preserve existing row-specific failures; never invent successful rows.
        path=out/'failures.csv'
        old=pd.read_csv(path) if path.exists() else pd.DataFrame()
        pd.concat([old,pd.DataFrame([{'stage':args.job,'cause':repr(exc)}])],ignore_index=True).to_csv(path,index=False)
        raise
