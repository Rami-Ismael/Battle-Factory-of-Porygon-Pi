import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000
def lin(c): return np.where(c<=.04045, c/12.92, ((c+.055)/1.055)**2.4)
M=np.array([[.4122214708,.5363325363,.0514459929],[.2119034982,.6806995451,.1073969566],[.0883024619,.2817188376,.6299787005]])
N=np.array([[.2104542553,.793617785,-.0040720468],[1.9779984951,-2.428592205,.4505937099],[.0259040371,.7827717662,-.808675766]])
def oklab(c):
    return np.cbrt(np.maximum(lin(c)@M.T,0))@N.T
def lab(c): return rgb2lab(c.reshape(-1,1,3)).reshape(-1,3)
rng=np.random.default_rng(0)
a=rng.random((400000,3)); b=rng.random((400000,3))
ok=100*np.linalg.norm(oklab(a)-oklab(b),axis=1); de=deltaE_ciede2000(lab(a),lab(b))
def q(x): return [round(float(v),1) for v in np.percentile(x,[5,50,95])]
s=(de>10.5)&(de<11.5); print("n",s.sum(),"OKLab x100 at CIEDE2000=11 (5/50/95):",q(ok[s]))
s=(ok>7.5)&(ok<8.5); print("n",s.sum(),"CIEDE2000 at OKLab x100=8:",q(de[s]))
s=(ok>14.5)&(ok<15.5); print("n",s.sum(),"CIEDE2000 at OKLab x100=15:",q(de[s]))
print("share at OKLab 8 reaching CIEDE2000 11:",round(float((de[(ok>7.5)&(ok<8.5)]>=11).mean()),3))
D=np.array([[.367322,.860646,-.227968],[.280085,.672501,.047413],[-.011820,.042940,.968881]])
def enc(l):
    l=np.clip(l,0,1); return np.where(l<=.0031308,12.92*l,1.055*l**(1/2.4)-.055)
def hx(h): return np.array([[int(h[i:i+2],16)/255 for i in (1,3,5)]])
for n,x,y in (("red/amber deutan","#c2392d","#a96f0c"),("green/orange deutan","#2f9e6b","#d4551f")):
    X=enc(lin(hx(x))@D.T);Y=enc(lin(hx(y))@D.T)
    print(n,"OKLab x100",round(float(100*np.linalg.norm(oklab(X)-oklab(Y))),1),"CIEDE2000",round(float(deltaE_ciede2000(lab(X),lab(Y))[0]),1))
