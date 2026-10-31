"""Issued-error buffer with complete canonical days and a conservative D-2 gate."""
from __future__ import annotations
from collections import deque
from datetime import timedelta
import numpy as np
import pandas as pd
from .data import LEVELS,NORMALIZED,day_hours,array_hash

BUFFER_POLICIES=('B0','B2','B3','A1','A2','A3','A4','A5')


class ResidualBuffer:
    def __init__(self):
        self.pending={};self.consumed=set();self.buffers={p:deque(maxlen=28) for p in BUFFER_POLICIES};self.trace=[];self.last_origin=None

    def issue(self,day,index,centers,scale):
        if day in self.pending or day in self.consumed:raise ValueError('issued day already registered')
        index=pd.DatetimeIndex(index).tz_convert('UTC')
        scale=np.asarray(scale,float)
        if set(centers)!=set(BUFFER_POLICIES) or len(index)!=len(scale) or not np.isfinite(scale).all() or (scale<=0).any():raise ValueError('invalid issued scale/policy schema')
        if any(np.asarray(v).shape!=(len(index),) or not np.isfinite(v).all() for v in centers.values()):raise ValueError('invalid issued center')
        self.pending[day]=(index,{k:np.asarray(v,float).copy() for k,v in centers.items()},scale.copy())

    def release(self,origin_day,truth):
        if self.last_origin is not None and origin_day<self.last_origin:
            raise ValueError('feedback origin cannot move backwards')
        self.last_origin=origin_day
        cutoff=origin_day-timedelta(days=2)
        for day in sorted(self.pending):
            if day>cutoff:break
            index,centers,scale=self.pending[day]
            if not index.equals(day_hours(day)):
                # Inherited exclusions mean no complete issued vector for this day.
                self.trace.append({'origin':str(origin_day),'feedback_day':str(day),'status':'incomplete_issued_day','n':len(index)})
                self.consumed.add(day);continue
            actual=np.asarray(truth(index),float)  # Only invoked AFTER the D-2 gate.
            if actual.shape!=(len(index),) or not np.isfinite(actual).all():
                self.trace.append({'origin':str(origin_day),'feedback_day':str(day),'status':'unavailable_complete_truth','n':0});continue
            for p in BUFFER_POLICIES:
                errors=(actual-centers[p])/(scale if p in NORMALIZED else 1.)
                # A previously unavailable day's truth can arrive after newer
                # days. Membership follows delivery date, not arrival order.
                latest=sorted([*self.buffers[p],(day,errors)],key=lambda item:item[0])[-28:]
                self.buffers[p]=deque(latest,maxlen=28)
                self.trace.append({'origin':str(origin_day),'feedback_day':str(day),'policy':p,'status':'consumed','n':len(errors),
                                   'issued_center_sha256':array_hash(centers[p]),'issued_scale_sha256':array_hash(scale),'error_sha256':array_hash(errors)})
            self.consumed.add(day)
        for d in self.consumed:self.pending.pop(d,None)

    def predict(self,policy,central,current_scale):
        buf=self.buffers[policy]
        if len(buf)!=28:raise ValueError(f'{policy}: need 28 complete released days, got {len(buf)}')
        errors=np.concatenate([x[1] for x in buf])
        quantiles=np.quantile(errors,LEVELS,method='linear')
        scale=np.asarray(current_scale) if policy in NORMALIZED else np.ones(len(central))
        q=np.asarray(central)[:,None]+scale[:,None]*quantiles
        if not np.isfinite(q).all() or (np.diff(q,axis=1)<0).any():raise ValueError('invalid emitted vector')
        return q,{'buffer_start':str(buf[0][0]),'buffer_end':str(buf[-1][0]),'buffer_days':28,'buffer_hours':len(errors),'buffer_sha256':array_hash(errors)}
