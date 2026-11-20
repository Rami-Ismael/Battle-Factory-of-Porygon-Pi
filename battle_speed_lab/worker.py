"""JSON-lines persistent BC worker; all policy changes are local subclasses."""
import asyncio
import hashlib
import io
import json
import os
import random
import resource
import sys
import time
from collections import Counter
from pathlib import Path

BENCH = Path.home() / '.local/share/vgc-pilot-runtime/unpacked/vgc-bench-d79f9532947ac114dce1dda2456a590afcd375b2'
sys.path.insert(0, str(BENCH))
import numpy as np
import torch
from poke_env.environment import DoublesEnv
from poke_env.concurrency import POKE_LOOP
from poke_env.player import DefaultBattleOrder
from poke_env.ps_client import AccountConfiguration, ServerConfiguration
from poke_env.teambuilder import Teambuilder
from stable_baselines3 import PPO
from vgc_bench.src.policy_player import BatchPolicyPlayer, _BatchReq
sys.path.insert(0, '/Users/ramiismael/Documents/code/vgc-team-generator-pilot/src')
from corpus import dex_entry


class SequenceTeams(Teambuilder):
    def __init__(self, pastes):
        self.teams = [self.join_team(self.parse_showdown_team(p)) for p in pastes]
        self.index = 0

    def yield_team(self):
        team = self.teams[self.index % len(self.teams)]
        self.index += 1
        return team


class Inference:
    def __init__(self, policy, config, metrics):
        self.policy, self.config, self.metrics = policy, config, metrics
        self.queue = asyncio.Queue()
        self.task = None
        self.pending = {}
        self.failure = None
        self.cache = None
        self.student = None
        self.remote = None
        if config.get('inference_socket'):
            if not config.get('actor_only') or config.get('student_file') or config.get('logit_cache'):
                raise ValueError('Remote actor requires actor_only with no student/cache')
            from remote_inference import RemoteActor
            self.remote=RemoteActor(config['inference_socket'])
        if config.get('student_file'):
            from apprentice import load_student
            self.student=load_student(config['student_file'])
        if config.get('logit_cache'):
            from inference import LogitCache
            self.cache = LogitCache()

    async def run(self):
        try:
            await self.run_loop()
        except Exception as exc:
            self.failure=exc
            for req in list(self.pending.values()):
                req.error=exc
                req.event.set()

    async def run_loop(self):
        while True:
            requests = [await self.queue.get()]
            mode = self.config.get('batcher', 'original')
            cap = self.config['concurrency']
            if self.config.get('shared'):
                cap *= 2
            if mode == 'original':
                just_slept = False
                while len(requests) < cap:
                    try:
                        requests.append(self.queue.get_nowait())
                        just_slept = False
                    except asyncio.QueueEmpty:
                        if just_slept:
                            break
                        await asyncio.sleep(.005)
                        just_slept = True
            else:
                deadline = requests[0][1] + self.config.get('wait_ms', 0) / 1000
                while len(requests) < cap:
                    try:
                        requests.append(self.queue.get_nowait())
                    except asyncio.QueueEmpty:
                        remaining = deadline - time.perf_counter()
                        if remaining <= 0:
                            break
                        try:
                            requests.append(await asyncio.wait_for(self.queue.get(), remaining))
                        except asyncio.TimeoutError:
                            break
            now = time.perf_counter()
            self.metrics['queue_seconds'].extend(now - t for _, t in requests)
            self.metrics['batch_sizes'].append(len(requests))
            reqs = [r for r, _ in requests]
            start = time.perf_counter()
            obs, masks = np.stack([r.obs for r in reqs]), np.stack([r.mask for r in reqs])
            self.metrics['stack_seconds'] += time.perf_counter() - start
            start = time.perf_counter()
            remote_logits=await self.remote.logits(obs,masks) if self.remote else None
            context = torch.inference_mode if self.config.get('inference_mode') else torch.no_grad
            with context():
                obs_dict = {
                    'observation': torch.as_tensor(obs, device=self.policy.device),
                    'action_mask': torch.as_tensor(masks, device=self.policy.device),
                }
                if self.config.get('actor_only') or self.config.get('per_battle_rng'):
                    from inference import actor_logits, sample_actions
                    if remote_logits is not None:
                        logits=torch.as_tensor(remote_logits,device=self.policy.device)
                    elif self.student is not None:
                        logits=self.student(obs_dict['observation'])
                    elif self.config.get('actor_only'):
                        logits = self.cache.compute(self.policy,obs_dict,obs,masks) if self.cache else actor_logits(self.policy,obs_dict)
                    else:
                        logits,_ = self.policy.get_logits(obs_dict,actor_grad=False)
                    if self.config.get('per_battle_rng'):
                        separate=[]
                        for i,req in enumerate(reqs):
                            with torch.random.fork_rng(devices=[]):
                                torch.set_rng_state(req.rng_state)
                                separate.append(sample_actions(self.policy,logits[i:i+1],obs_dict['action_mask'][i:i+1],False))
                                req.rng_state = torch.get_rng_state()
                        actions = torch.cat(separate)
                    else:
                        actions = sample_actions(self.policy,logits,obs_dict['action_mask'],self.config.get('deterministic',False))
                else:
                    actions, _, _ = self.policy.forward(obs_dict, deterministic=self.config.get('deterministic', False))
            actions = actions.cpu().numpy()
            self.metrics['forward_seconds'] += time.perf_counter() - start
            for req, action in zip(reqs, actions):
                req.result = action
                req.event.set()

    async def close(self):
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        if self.remote:await self.remote.close()


class InstrumentedPlayer(BatchPolicyPlayer):
    def __init__(self, service, metrics, **kwargs):
        self.service, self.metrics = service, metrics
        self._rng_by_battle = {}
        super().__init__(**kwargs)

    async def _choose_move(self, battle):
        if battle._wait:
            return DefaultBattleOrder()
        start = time.perf_counter()
        if self.service.config.get('encoder') == 'fast':
            from encoder import embed_battle
            obs = embed_battle(battle, fake_rating=2000)
            if self.service.config.get('verify_encoder'):
                reference = self.embed_battle(battle, fake_rating=2000)
                if not np.array_equal(obs, reference):
                    raise AssertionError('Optimized encoder changed observation values')
                self.metrics['encoder_verified'] = self.metrics.get('encoder_verified',0)+1
        else:
            obs = self.embed_battle(battle, fake_rating=2000)
        self.metrics['embedding_seconds'] += time.perf_counter() - start
        start = time.perf_counter()
        mask = np.array(DoublesEnv.get_action_mask(battle))
        self.metrics['mask_seconds'] += time.perf_counter() - start
        if self.service.config.get('capture'):
            self.metrics['capture_seen'] = self.metrics.get('capture_seen',0)+1
            samples = self.metrics.setdefault('_samples',[])
            stride=self.service.config.get('capture_stride',4)
            limit=self.service.config.get('capture_limit',512)
            if self.metrics['capture_seen']%stride==0:
                seen=self.metrics['capture_seen']//stride
                slot=len(samples) if len(samples)<limit else self.metrics['_capture_rng'].randrange(seen)
                if slot<limit:
                    sample=(obs.copy(),mask.copy())
                    identity=(battle.battle_tag,battle.turn,self.username)
                    ids=self.metrics.setdefault('_sample_ids',[])
                    if slot==len(samples):samples.append(sample);ids.append(identity)
                    else:samples[slot]=sample;ids[slot]=identity
        req = _BatchReq(obs=obs, mask=mask, event=asyncio.Event())
        if self.service.failure is not None:
            raise RuntimeError('Inference service failed') from self.service.failure
        self.service.pending[id(req)]=req
        if self.service.config.get('per_battle_rng'):
            if battle.battle_tag not in self._rng_by_battle:
                key=f"{self.service.config.get('policy_seed',20261013)}:{battle.battle_tag}:{self.username}"
                seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'big')%(2**63)
                generator=torch.Generator().manual_seed(seed)
                self._rng_by_battle[battle.battle_tag]=generator.get_state()
            req.rng_state=self._rng_by_battle[battle.battle_tag]
        if self.service.task is None:
            self.service.task = asyncio.create_task(self.service.run())
        await self.service.queue.put((req, time.perf_counter()))
        await req.event.wait()
        self.service.pending.pop(id(req),None)
        if getattr(req,'error',None) is not None:
            raise RuntimeError('Inference request failed') from req.error
        if self.service.config.get('per_battle_rng'):
            self._rng_by_battle[battle.battle_tag]=req.rng_state
        return DoublesEnv.action_to_order(req.result, battle)


async def trial(policy, request):
    config = request['config']
    seed = request['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    metrics = {'embedding_seconds': 0., 'mask_seconds': 0., 'forward_seconds': 0.,
               'stack_seconds': 0., 'queue_seconds': [], 'batch_sizes': []}
    if config.get('capture'):
        if config.get('capture_stride',4)<1 or config.get('capture_limit',512)<1:
            raise ValueError('Capture stride and limit must be positive')
        metrics['_capture_rng']=random.Random(request.get('seed',0)+700001)
    services = [Inference(policy, config, metrics)]
    if not config.get('shared'):
        services.append(Inference(policy, config, metrics))
    srv = ServerConfiguration(f"ws://127.0.0.1:{request['port']}/showdown/websocket",
                              'https://play.pokemonshowdown.com/action.php?')
    kw = dict(policy=policy, server_configuration=srv, battle_format=request['format'],
              max_concurrent_battles=config['concurrency'], log_level=50,
              accept_open_team_sheet=True, open_timeout=30,
              deterministic=config.get('deterministic', False))
    tag = os.urandom(5).hex()
    pairs = request['pairs']
    def make_player(service,name,pastes):
        args=dict(account_configuration=AccountConfiguration(name,None),team=SequenceTeams(pastes),**kw)
        if config.get('upstream'):
            return BatchPolicyPlayer(**args)
        return InstrumentedPlayer(service,metrics,**args)
    p1 = make_player(services[0],'a'+tag,[x['a'] for x in pairs])
    p2 = make_player(services[-1],'b'+tag,[x['b'] for x in pairs])
    cpu0 = resource.getrusage(resource.RUSAGE_SELF)
    start = time.perf_counter()
    try:
        await asyncio.wait_for(p1.battle_against(p2, n_battles=len(pairs)), request.get('timeout', 180))
        battles = list(p1.battles.values())
        records = [{'tag': b.battle_tag, 'finished': b.finished, 'turns': b.turn, 'won': b.won,
                    'team': sorted(m.species for m in b.teampreview_team),
                    'opponent': sorted(m.species for m in b.teampreview_opponent_team)} for b in battles]
        if len(battles) != len(pairs) or not all(b.finished for b in battles):
            raise RuntimeError('Incomplete battle workload')
        norm = lambda x: ''.join(c for c in x.lower() if c.isalnum())
        def base(species):
            entry = dex_entry(species)
            return norm(entry.get('baseSpecies',species) if entry else species)
        expected = Counter((p['a_base'],p['b_base']) for p in pairs)
        observed = Counter((','.join(sorted(base(x) for x in r['team'])),
                            ','.join(sorted(base(x) for x in r['opponent']))) for r in records)
        if observed != expected:
            raise RuntimeError('Completed team matchups do not match the requested workload: '+str(observed-expected))
        result = {'seconds': time.perf_counter()-start, 'completed': len(battles),
                  'turns': sum(b.turn for b in battles), 'wins': sum(b.won is True for b in battles),
                  'ties': sum(b.won is None for b in battles), 'battles': records,
                  'attribution_verified': True, 'attribution_method':'base-species preview multiset; fixed candidate'}
    finally:
        for p in (p1, p2):
            await p.ps_client.stop_listening()
            if config.get('upstream') and p._worker_task is not None:
                p._worker_task.cancel()
                await asyncio.gather(p._worker_task,return_exceptions=True)
        for service in services:
            await service.close()
    cpu1 = resource.getrusage(resource.RUSAGE_SELF)
    result['python_cpu_seconds'] = cpu1.ru_utime+cpu1.ru_stime-cpu0.ru_utime-cpu0.ru_stime
    result['python_peak_rss_bytes'] = cpu1.ru_maxrss
    waits = metrics.pop('queue_seconds')
    sizes = metrics.pop('batch_sizes')
    metrics.update({'queue_mean_ms': float(np.mean(waits)*1000) if waits else 0,
                    'queue_p95_ms': float(np.percentile(waits, 95)*1000) if waits else 0,
                    'batch_mean': float(np.mean(sizes)) if sizes else 0,
                    'inference_calls': len(sizes), 'batch_histogram': dict(Counter(sizes))})
    result['metrics'] = metrics
    samples = metrics.pop('_samples',[])
    sample_ids = metrics.pop('_sample_ids',[])
    metrics.pop('_capture_rng',None)
    if samples:
        path = Path(request['capture_path'])
        np.savez_compressed(path,observations=np.stack([o for o,m in samples]),masks=np.stack([m for o,m in samples]),
                            battle_ids=np.asarray([x[0] for x in sample_ids]),
                            turns=np.asarray([x[1] for x in sample_ids]),sides=np.asarray([x[2] for x in sample_ids]))
        result['capture'] = {'path':str(path),'observations':len(samples)}
    caches = [s.cache for s in services if s.cache]
    if caches:
        metrics['logit_cache_hits'] = sum(c.hits for c in caches)
        metrics['logit_cache_misses'] = sum(c.misses for c in caches)
    return result


def main():
    config = json.loads(sys.argv[1])
    torch.set_num_threads(config.get('threads', 8))
    torch.set_num_interop_threads(1)
    start = time.perf_counter()
    # Load the exact bytes verified here, even if another process replaces the path.
    checkpoint = Path(config.get('checkpoint_path', '/tmp/bc_100.zip')).read_bytes()
    if config.get('checkpoint_sha256') and hashlib.sha256(checkpoint).hexdigest() != config['checkpoint_sha256']:
        raise ValueError('Worker checkpoint differs from the requested policy')
    policy = PPO.load(io.BytesIO(checkpoint), device=torch.device('cpu')).policy
    loaded_training=policy.training
    if not config.get('preserve_training'):
        policy.set_training_mode(False)
    print(json.dumps({'ready': True, 'load_seconds': time.perf_counter()-start,
                      'threads': torch.get_num_threads(), 'torch': torch.__version__,
                      'loaded_training':loaded_training,'effective_training':policy.training}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            result = asyncio.run_coroutine_threadsafe(trial(policy, request), POKE_LOOP).result()
            print(json.dumps(result), flush=True)
        except Exception as exc:
            import traceback
            traceback.print_exc(file=sys.stderr)
            print(json.dumps({'error': type(exc).__name__+': '+str(exc)}), flush=True)
            return


if __name__ == '__main__':
    main()
