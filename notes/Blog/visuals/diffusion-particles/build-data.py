"""Seeded OU forward diffusion and analytic-score reverse probability-flow ODE.
Target: equal mixture of three isotropic 2D Gaussians, variance 0.10.
Forward dX=-X/2 dt+dW. Reverse integrates the probability-flow ODE backwards.
This is a continuous toy, not a trace from the categorical team generator.
"""
import math,json,random
from pathlib import Path
r=random.Random(1826); N=100; steps=240; T=6.;dt=T/steps
mu=[(-1.65,-1.05),(1.65,-1.05),(0,1.8)];var=.10
initial=[[mu[i%3][j]+r.gauss(0,math.sqrt(var)) for j in range(2)] for i in range(N)]
fwd=[initial]
a=math.exp(-dt/2);noise=math.sqrt(1-math.exp(-dt))
for _ in range(steps):fwd.append([[a*x+noise*r.gauss(0,1) for x in p] for p in fwd[-1]])
# Start reverse from the same terminal draws. Marginal is p_T, not asserted exactly N(0,I).
def velocity(x,t):
 a=math.exp(-t/2);v=a*a*var+1-a*a
 means=[[a*z for z in m] for m in mu]
 logs=[-sum((x[j]-m[j])**2 for j in range(2))/(2*v) for m in means];mx=max(logs)
 weights=[math.exp(l-mx) for l in logs];total=sum(weights)
 score=[sum(w*(m[j]-x[j])/v for w,m in zip(weights,means))/total for j in range(2)]
 return [-.5*x[j]-.5*score[j] for j in range(2)]
rev=[fwd[-1]]
for k in range(steps):
 t=T-k*dt;row=[]
 for x in rev[-1]:
  v=velocity(x,t);mid=[x[j]-dt*v[j]/2 for j in range(2)];vm=velocity(mid,t-dt/2)
  row.append([x[j]-dt*vm[j] for j in range(2)])
 rev.append(row)
# Store every other frame; interpolation is only for display.
data={'forward':[[[round(z,4) for z in p] for p in row] for row in fwd[::2]],'reverse':[[[round(z,4) for z in p] for p in row] for row in rev[::2]],'means':mu,'T':T,'particles':N}
Path(__file__).with_name('paths.json').write_text(json.dumps(data,separators=(',',':')))
print('Generated forward SDE and reverse ODE paths:',len(data['forward']),'frames;',N,'particles')
