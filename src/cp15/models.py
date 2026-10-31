"""Fresh daily LEAR and LightGBM fits; no access to future targets in fit indices."""
from __future__ import annotations
from datetime import date,timedelta
import time,warnings
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from .data import Inputs, array_hash, history_start, origin_utc


def prepared_linear(train,other):
    # Explicit fixed-dimensional missing indicators, learned imputation on train only.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        fill=np.nanmedian(train,axis=0)
    fill=np.where(np.isfinite(fill),fill,0.)
    def transform(x):return np.column_stack((np.where(np.isfinite(x),x,fill),~np.isfinite(x))).astype(float)
    scaler=StandardScaler().fit(transform(train))
    return scaler.transform(transform(train)),scaler.transform(transform(other)),fill,scaler


def _lasso(x,y,alpha,p,warm=None):
    model=warm if warm is not None else Lasso(max_iter=p['max_iter'],tol=p['tol'],selection=p['selection'],warm_start=True)
    model.alpha=alpha
    # Continue the SAME optimization at the SAME alpha/tolerance if a 20,000-
    # iteration solver call is exhausted. Never accept an unconverged endpoint.
    # This numerical repair is recorded before any outer evaluation result.
    total=0
    for attempt in range(10):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            model.fit(x,y)
        total+=int(model.n_iter_)
        if not any(issubclass(w.category,ConvergenceWarning) for w in caught):
            model.solver_passes_=attempt+1;model.total_n_iter_=total
            return model
    raise RuntimeError(f'LEAR convergence failure after 10 fixed-budget continuations: alpha={alpha}, shape={x.shape}, dual_gap={model.dual_gap_}')


def fit_day(data: Inputs, d: date, policy: str, *, rows=None):
    started=time.perf_counter()
    lower=history_start(d,policy)
    train=np.flatnonzero((data.dates>=np.datetime64(lower))&(data.dates<np.datetime64(d))&data.eligible)
    rows=data.rows(d) if rows is None else np.asarray(rows)
    if not len(rows):return np.array([]),[]
    norm=policy.startswith('A')
    target=(data.y-data.level)/data.scale if norm else data.y
    log=[]
    base={'policy':policy,'delivery_date':str(d),'origin_utc':str(origin_utc(d).tz_convert('UTC')),
          'history_start':str(lower),'history_end_exclusive':str(d),'n_train':len(train),
          'train_rows_sha256':array_hash(train),'train_target_sha256':array_hash(target[train]),
          'normalization_rows_sha256':array_hash(np.column_stack((data.level[train],data.scale[train]))),
          'latest_training_delivery_date':str(data.dates[train].max()) if len(train) else None}
    if policy in ('B3','A2'):
        if len(train)<data.p['history']['minimum_lgbm_training_rows']:raise ValueError('LGBM training insufficiency')
        x=data.lgbm_normalized if norm else data.lgbm_raw
        params=dict(data.p['lgbm']['parameters'])
        model=LGBMRegressor(alpha=.5,random_state=data.p['seed'],**params)
        model.fit(x[train],target[train])
        pred=model.predict(x[rows])
        log.append({**base,'local_hour':-1,'fit_calls':1,'validation_rows':0,'fit_seconds':time.perf_counter()-started,
                    'model_sha256':__import__('hashlib').sha256(model.booster_.model_to_string().encode()).hexdigest()})
    else:
        x=data.lear_normalized if norm else data.lear_raw
        pred=np.empty(len(rows));p=data.p['lear']
        split=d-timedelta(days=28)
        for hour in np.unique(data.hours[rows]):
            hs=time.perf_counter();tr=train[data.hours[train]==hour]
            required=data.p['history']['minimum_lear_rows_per_hour_short' if policy=='A4' else 'minimum_lear_rows_per_hour_long']
            if len(tr)<required:raise ValueError(f'LEAR training insufficiency {policy} {d} hour {hour}: {len(tr)} < {required}')
            inner=tr[data.dates[tr]<np.datetime64(split)];val=tr[data.dates[tr]>=np.datetime64(split)]
            if len(inner)<data.p['history']['minimum_lear_inner_train_rows_per_hour'] or len(val)<data.p['history']['minimum_lear_validation_rows_per_hour']:raise ValueError('LEAR chronological validation insufficiency')
            it,iv,_,_=prepared_linear(x[inner],x[val]);unit=max(float(np.std(target[inner])),1e-8)
            losses={};model=None;iterations={};passes={}
            for rel in reversed(p['relative_alpha_grid']):
                model=_lasso(it,target[inner],rel*unit,p,model)
                vp=model.predict(iv)
                if norm:vp=vp*data.scale[val]+data.level[val]
                losses[rel]=float(np.mean(np.abs(vp-data.y[val])));iterations[rel]=int(model.total_n_iter_);passes[rel]=int(model.solver_passes_)
            selected=min(p['relative_alpha_grid'],key=lambda rel:losses[rel])
            selected_rows=rows[data.hours[rows]==hour]
            xt,xp,fill,scaler=prepared_linear(x[tr],x[selected_rows])
            finalunit=max(float(np.std(target[tr])),1e-8)
            final=_lasso(xt,target[tr],selected*finalunit,p)
            pred[data.hours[rows]==hour]=final.predict(xp)
            log.append({**base,'local_hour':int(hour),'n_hour_train':len(tr),'inner_train_rows':len(inner),'validation_rows':len(val),
                        'inner_train_end':str(data.dates[inner].max()),'validation_start':str(data.dates[val].min()),'validation_end':str(data.dates[val].max()),
                        'selected_relative_alpha':selected,'alpha':selected*finalunit,'validation_mae_by_alpha':{str(k):v for k,v in losses.items()},
                        'n_iter_by_alpha':{str(k):v for k,v in iterations.items()},'final_n_iter':int(final.total_n_iter_),
                        'solver_passes_by_alpha':{str(k):v for k,v in passes.items()},'final_solver_passes':int(final.solver_passes_),
                        'solver_calls':sum(passes.values())+int(final.solver_passes_),'final_dual_gap':float(final.dual_gap_),
                        'hour_train_rows_sha256':array_hash(tr),'inner_train_rows_sha256':array_hash(inner),'validation_rows_sha256':array_hash(val),
                        'imputer_sha256':array_hash(fill),'scaler_sha256':array_hash(np.column_stack((scaler.mean_,scaler.scale_))),
                        'model_sha256':array_hash(np.r_[final.coef_,final.intercept_]),'fit_calls':len(p['relative_alpha_grid'])+1,'fit_seconds':time.perf_counter()-hs})
    if norm:pred=pred*data.scale[rows]+data.level[rows]
    if not np.isfinite(pred).all():raise ValueError('nonfinite model forecast')
    return pred,log
