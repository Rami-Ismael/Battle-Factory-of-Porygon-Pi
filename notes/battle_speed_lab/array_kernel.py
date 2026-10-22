"""A restricted fixed-damage doubles kernel, not a general Pokemon simulator.

Exactly four level-50 Chansey, Natural Cure, no items, Seismic Toss only, no
switches/status/weather/field effects/tera, unique unmodified speeds. Both sides
target the first living opponent. Unsupported battles must stay in Showdown.
"""
import numpy as np


def eligible(team):
    if len(team)!=2:return False
    for pokemon in team:
        if pokemon.get('species')!='Chansey' or pokemon.get('ability')!='Natural Cure':return False
        if pokemon.get('item') or pokemon.get('level')!=50:return False
        if pokemon.get('moves')!=['Seismic Toss']:return False
    return True


def simulate(initial_hp,speeds,record_events=False):
    hp=np.asarray(initial_hp,dtype=np.int64).copy();speed=np.asarray(speeds,dtype=np.int64)
    if hp.ndim!=2 or hp.shape[1]!=4 or speed.shape!=hp.shape:raise ValueError('Expected B x 4 states')
    if np.any(hp<=0) or np.any(hp>400):raise ValueError('Unsupported HP range')
    if np.any(np.diff(np.sort(speed,axis=1),axis=1)==0):raise ValueError('Speed ties require reference RNG')
    order=np.argsort(-speed,axis=1);batch=np.arange(len(hp));turns=np.zeros(len(hp),dtype=int)
    done=np.zeros(len(hp),dtype=bool);events=[[] for _ in hp] if record_events else None
    for turn in range(1,33):
        if done.all():break
        turns[~done]=turn
        for position in range(4):
            actor=order[:,position]
            live=(~done)&(hp[batch,actor]>0)
            indices=batch[live]
            base=np.where(actor[live]<2,2,0)
            target=base+(hp[indices,base]<=0).astype(int)
            hp[indices,target]=np.maximum(0,hp[indices,target]-50)
            if record_events:
                for i,t in zip(indices,target):events[i].append([turn,int(t),int(hp[i,t])])
            done=(hp[:,:2].sum(1)==0)|(hp[:,2:].sum(1)==0)
    if not done.all():raise RuntimeError('Restricted PP/turn limit exceeded; no approximate result returned')
    winner=np.where(hp[:,:2].sum(1)>0,0,1)
    return {'hp':hp,'turns':turns,'winner':winner,'events':events}


def scalar(initial_hp,speeds,record_events=False):
    """Same restricted rules, individual Python battles: vectorization control."""
    output=[];turns=[];winners=[];all_events=[]
    for initial,speed in zip(initial_hp,speeds):
        hp=list(map(int,initial));order=sorted(range(4),key=lambda i:-speed[i]);events=[]
        if len(hp)!=4 or len(set(map(int,speed)))!=4 or min(hp)<=0 or max(hp)>400:
            raise ValueError('Unsupported restricted state')
        for turn in range(1,33):
            for actor in order:
                if hp[actor]<=0:continue
                base=2 if actor<2 else 0;target=base if hp[base]>0 else base+1
                hp[target]=max(0,hp[target]-50)
                if record_events:events.append([turn,target,hp[target]])
                if not sum(hp[:2]) or not sum(hp[2:]):break
            if not sum(hp[:2]) or not sum(hp[2:]):break
        else:raise RuntimeError('Restricted PP/turn limit exceeded')
        output.append(hp);turns.append(turn);winners.append(0 if sum(hp[:2]) else 1);all_events.append(events)
    return {'hp':np.asarray(output),'turns':np.asarray(turns),'winner':np.asarray(winners),
            'events':all_events if record_events else None}
