"""Local cross-worker actor batching prototype.

Only logits cross the socket. Masks, conditional action sampling and per-battle
RNG remain with the caller. Workers still load their policy for that reference
sampling logic, so this prototype does not claim to remove model duplication.
The transport uses length-delimited NumPy arrays, never pickle.
"""
import asyncio
import io
import json
import struct
import time

MAX_FRAME=64*1024*1024


async def read_frame(reader):
    size=struct.unpack('!I',await reader.readexactly(4))[0]
    if size>MAX_FRAME:raise ValueError('Inference frame exceeds limit')
    return await reader.readexactly(size)


async def write_frame(writer,payload):
    if len(payload)>MAX_FRAME:raise ValueError('Inference frame exceeds limit')
    writer.write(struct.pack('!I',len(payload))+payload)
    await writer.drain()


class RemoteActor:
    def __init__(self,socket_path):
        self.path=socket_path
        self.reader=self.writer=None

    async def logits(self,observations,masks):
        import numpy as np
        if self.writer is None:
            self.reader,self.writer=await asyncio.open_unix_connection(self.path)
        payload=io.BytesIO()
        np.savez(payload,observations=observations,masks=masks)
        await write_frame(self.writer,payload.getvalue())
        response=await asyncio.wait_for(read_frame(self.reader),60)
        if response[:1]!=b'\x00':raise RuntimeError(response[1:].decode())
        result=np.load(io.BytesIO(response[1:]),allow_pickle=False)
        if result.ndim!=2 or result.shape[0]!=len(observations):
            raise ValueError('Remote actor returned wrong batch shape')
        return result

    async def close(self):
        if self.writer is not None:
            self.writer.close()
            await self.writer.wait_closed()
            self.reader=self.writer=None


async def serve(policy,socket_path,wait_ms,max_batch,stats_path):
    import numpy as np
    import torch
    from inference import actor_logits
    queue=asyncio.Queue()
    stats={'batches':0,'requests':0,'observations':0,'batch_sizes':[],
           'forward_seconds':0.,'wait_ms':wait_ms,'max_batch':max_batch}

    async def connection(reader,writer):
        try:
            while True:
                payload=await read_frame(reader)
                with np.load(io.BytesIO(payload),allow_pickle=False) as arrays:
                    obs,mask=arrays['observations'],arrays['masks']
                if obs.ndim!=2 or mask.ndim!=2 or len(obs)!=len(mask) or not len(obs):
                    raise ValueError('Invalid observation/mask dimensions')
                future=asyncio.get_running_loop().create_future()
                await queue.put((obs,mask,future,time.perf_counter()))
                result=await future
                output=io.BytesIO();np.save(output,result,allow_pickle=False)
                await write_frame(writer,b'\x00'+output.getvalue())
        except asyncio.IncompleteReadError:
            pass
        except Exception as exc:
            try:await write_frame(writer,b'\x01'+str(exc).encode())
            except (ConnectionError,BrokenPipeError):pass
        finally:
            writer.close()
            await writer.wait_closed()

    async def batching():
        while True:
            requests=[await queue.get()]
            deadline=requests[0][3]+wait_ms/1000
            count=len(requests[0][0])
            while count<max_batch:
                try:request=queue.get_nowait()
                except asyncio.QueueEmpty:
                    remaining=deadline-time.perf_counter()
                    if remaining<=0:break
                    try:request=await asyncio.wait_for(queue.get(),remaining)
                    except asyncio.TimeoutError:break
                requests.append(request);count+=len(request[0])
            try:
                obs=np.concatenate([r[0] for r in requests])
                masks=np.concatenate([r[1] for r in requests])
                start=time.perf_counter()
                # The cap is also enforced on actual model batches when a whole
                # client request crosses the gathering threshold.
                with torch.inference_mode():
                    logits=torch.cat([actor_logits(policy,{
                        'observation':torch.as_tensor(obs[i:i+max_batch]),
                        'action_mask':torch.as_tensor(masks[i:i+max_batch])})
                        for i in range(0,len(obs),max_batch)]).numpy()
                stats['forward_seconds']+=time.perf_counter()-start
                stats['batches']+=1;stats['requests']+=len(requests)
                stats['observations']+=count;stats['batch_sizes'].append(count)
                offset=0
                for observations,_,future,_ in requests:
                    n=len(observations)
                    if not future.done():future.set_result(logits[offset:offset+n])
                    offset+=n
            except Exception as exc:
                for _,_,future,_ in requests:
                    if not future.done():future.set_exception(exc)

    server=await asyncio.start_unix_server(connection,path=socket_path)
    worker=asyncio.create_task(batching())
    print(json.dumps({'ready':True,'socket':socket_path}),flush=True)
    try:
        async with server:await server.serve_forever()
    finally:
        worker.cancel();await asyncio.gather(worker,return_exceptions=True)
        if stats_path:stats_path.write_text(json.dumps(stats,indent=2)+'\n')


def main():
    import argparse
    import os
    import signal
    import sys
    from pathlib import Path
    ap=argparse.ArgumentParser()
    ap.add_argument('--socket',required=True)
    ap.add_argument('--wait-ms',type=float,default=1.)
    ap.add_argument('--max-batch',type=int,default=192)
    ap.add_argument('--stats',type=Path)
    args=ap.parse_args()
    if args.stats:args.stats=args.stats.resolve()
    if args.wait_ms<0 or args.max_batch<1:ap.error('Invalid batching parameters')
    if Path(args.socket).exists():ap.error('Refusing to replace an existing socket path')
    bench=Path.home()/'.local/share/vgc-pilot-runtime/unpacked/vgc-bench-d79f9532947ac114dce1dda2456a590afcd375b2'
    sys.path.insert(0,str(bench));os.chdir(bench)
    import torch
    from stable_baselines3 import PPO
    from vgc_bench.src import policy as _policy
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    policy=PPO.load('/tmp/bc_100.zip',device='cpu').policy
    policy.set_training_mode(False)
    async def run():
        task=asyncio.current_task()
        asyncio.get_running_loop().add_signal_handler(signal.SIGTERM,task.cancel)
        try:await serve(policy,args.socket,args.wait_ms,args.max_batch,args.stats)
        except asyncio.CancelledError:pass
    asyncio.run(run())


if __name__=='__main__':main()
