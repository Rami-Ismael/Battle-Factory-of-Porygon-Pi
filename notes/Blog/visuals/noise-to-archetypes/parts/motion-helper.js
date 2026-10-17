/* Motion helper. Tools only; motion.js decides what animates.
   Motion.reduced            -> true when prefers-reduced-motion: reduce.
   Motion.attach(setup)      -> setup(ctx) runs after figInit; it returns onState(state), called after every render (ctx.show).
   Motion.flip(root, selector, opts) -> registers FLIP for re-rendered marks: after each render, every element matching
        selector inside root slides from where the previous one with the same key was ON SCREEN (mid-flight positions included,
        so a second click retargets instead of jumping) to its new spot. Uses transform only (WAAPI), never on reduced motion.
        opts: {key:(el,i)=>string (default el.dataset.k || i), duration:ms, easing:css, enter:'fade'|null (opacity 0->1 for new keys),
               enterDuration:ms}. Do not FLIP elements that carry their own SVG transform attribute (e.g. a rotated ellipse):
               wrap them in a <g> or animate their opacity instead.
   Motion.fadeIn(el, ms, easing) / Motion.fadeOut(el, ms, easing) -> opacity only (allowed under reduced motion).
   Motion.EASE = {out:'cubic-bezier(0.23, 1, 0.32, 1)', inOut:'cubic-bezier(0.77, 0, 0.175, 1)', drawer:'cubic-bezier(0.32, 0.72, 0, 1)'}
*/
(function(){
 const mq=window.matchMedia('(prefers-reduced-motion: reduce)');
 const M={get reduced(){return mq.matches},EASE:{out:'cubic-bezier(0.23, 1, 0.32, 1)',inOut:'cubic-bezier(0.77, 0, 0.175, 1)',drawer:'cubic-bezier(0.32, 0.72, 0, 1)'}};
 const flips=[];const hooks=[];
 const centre=el=>{const r=el.getBoundingClientRect();return [r.left+r.width/2,r.top+r.height/2]};
 /* snapshot visual positions just before any reader action re-renders */
 const snap=()=>{for(const f of flips){f.prev=new Map();const els=f.root.querySelectorAll(f.sel);els.forEach((el,i)=>f.prev.set(f.key(el,i),centre(el)))}};
 ['pointerdown','keydown','input','change','scroll','wheel'].forEach(t=>window.addEventListener(t,snap,{capture:true,passive:true}));
 M.flip=(root,sel,o={})=>{const f={root:typeof root==='string'?document.querySelector(root):root,sel,key:o.key||((el,i)=>el.dataset.k||String(i)),dur:o.duration||240,ease:o.easing||M.EASE.inOut,enter:o.enter||null,enterDur:o.enterDuration||(o.duration||240),prev:null};
  flips.push(f);return f};
 const play=f=>{if(!f.root)return;const els=f.root.querySelectorAll(f.sel);const svg=f.root.closest('svg')||f.root.ownerSVGElement;
  const k=svg&&svg.getScreenCTM?svg.getScreenCTM().a:1;
  els.forEach((el,i)=>{const key=f.key(el,i);const old=f.prev&&f.prev.get(key);
   if(!old){if(f.enter==='fade'&&f.prev)el.animate([{opacity:0},{opacity:1}],{duration:f.enterDur,easing:M.EASE.out});return}
   if(M.reduced)return;
   const now=centre(el);const dx=(old[0]-now[0])/(svg?k:1),dy=(old[1]-now[1])/(svg?k:1);
   if(Math.abs(dx)+Math.abs(dy)<0.5)return;
   el.getAnimations().forEach(a=>a.cancel());
   el.animate([{transform:`translate(${dx}px, ${dy}px)`},{transform:'translate(0px, 0px)'}],{duration:f.dur,easing:f.ease})});
  f.prev=null};
 M.fadeIn=(el,ms=200,e=M.EASE.out)=>el&&el.animate([{opacity:0},{opacity:1}],{duration:ms,easing:e});
 M.fadeOut=(el,ms=150,e=M.EASE.out)=>el&&el.animate([{opacity:1},{opacity:0}],{duration:ms,easing:e});
 M.attach=setup=>hooks.push(setup);
 M._wrap=()=>{const orig=window.figInit;if(typeof orig!=='function')return;
  window.figInit=function(ctx){const show=ctx.show;let fns=[];
   ctx.show=function(s){show(s);flips.forEach(play);fns.forEach(fn=>{try{fn(s)}catch(e){console.error(e)}})};
   orig(ctx);fns=hooks.map(h=>{try{return h(ctx)}catch(e){console.error(e);return null}}).filter(Boolean)}};
 window.Motion=M;
})();
