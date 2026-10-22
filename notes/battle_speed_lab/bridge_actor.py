"""Route reference-encoded observations to the actor inside the Node bridge."""
import asyncio
import base64
import numpy as np


class BridgeActor:
    def __init__(self,send,futures):self.send=send;self.futures=futures;self.next_id=0
    async def logits(self,observations,masks):
        identity='inference-'+str(self.next_id);self.next_id+=1
        future=asyncio.get_running_loop().create_future();self.futures[identity]=future
        try:
            values=np.asarray(observations,dtype=np.float32)
            await self.send({'kind':'infer','id':identity,'shape':list(values.shape),
                             'data':base64.b64encode(values.tobytes()).decode()})
            result=await asyncio.wait_for(future,60)
            if result['shape']!=[len(values),masks.shape[1]]:raise ValueError('Incorrect Node actor output shape')
            return np.frombuffer(base64.b64decode(result['data']),dtype=np.float32).reshape(result['shape']).copy()
        finally:self.futures.pop(identity,None)
    async def close(self):
        for future in list(self.futures.values()):
            if not future.done():future.cancel()
