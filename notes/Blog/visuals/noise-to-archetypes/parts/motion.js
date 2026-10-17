/* Motion layer: the cloud remembers where it was.
   One-step scrolls and one-step range drags stay instant. A chapter press, or a range
   jump of more than one step, flows the 300 team dots and MB607's mean from the positions
   they had before the render (snapshot taken on the reader action) to the new ones, and
   crossfades the futures. Scroll events are only snapshotted inside a flow window.
   The team sheet fades in over 200 ms and out over 120 ms; a reopen during the exit
   takes over from the current opacity instead of stacking a second fade. */
Motion.attach(function(ctx){
  const noise=document.getElementById('sc-noise'), mb=document.getElementById('sc-mb');
  const range=document.getElementById('sc-step');
  const sheet=document.getElementById('sc-sheet');
  const REVEALS=['sc-reveal-1','sc-reveal-2','sc-reveal-3','sc-reveal-4','sc-tally'].map(id=>document.getElementById(id)).filter(Boolean);
  const SVGNS='http://www.w3.org/2000/svg';
  const DOT_FLOW=true; /* false: keep the futures crossfade and the MB dot slide, drop the 300-dot flow */
  const FLOW_MS=240, FADE_IN_MS=200, FADE_OUT_MS=180, SHEET_OUT_MS=120, WINDOW_MS=500;
  const was=new Map();
  let snap=null, prev=null, pressAt=-1e9, flowUntil=-1e9, rangeHit=false;

  const centre=el=>{const r=el.getBoundingClientRect();return [r.left+r.width/2,r.top+r.height/2]};
  const pts=(root,sel)=>root?Array.from(root.querySelectorAll(sel),el=>centre(el)):[];

  /* snapshot before the render: positions (mid-flight included) and copies of the futures to crossfade */
  function takeSnap(){
    const first=mb?mb.querySelector('.sc-fut, .sc-ell-mb'):null;
    snap={dots:pts(noise,'.sc-dot'),mbDot:pts(mb,'.sc-mb-dot'),mbRing:pts(mb,'.sc-mb-ring'),
      ghost:mb?Array.from(mb.querySelectorAll('.sc-fut, .sc-ell-mb'),n=>n.cloneNode()):[],
      ghostT:first&&first.parentNode.getAttribute?first.parentNode.getAttribute('transform'):null};
  }
  const onAct=e=>{const t=e.target;if(!t||!t.closest)return;
    if(t.closest('.sc-chap')){pressAt=performance.now();takeSnap();}
    else if(t.closest('#sc-step'))takeSnap();};
  window.addEventListener('pointerdown',onAct,{capture:true,passive:true});
  window.addEventListener('keydown',onAct,{capture:true,passive:true});
  window.addEventListener('input',e=>{if(e.target&&e.target.id==='sc-step'){rangeHit=true;takeSnap();}},{capture:true,passive:true});
  window.addEventListener('scroll',()=>{if(performance.now()<flowUntil)takeSnap();},{capture:true,passive:true});

  /* each element that kept its index starts at its old centre and eases to its new one (transform only) */
  function slide(root,sel,old,ms){
    if(!root||!old)return;
    const k=root.getScreenCTM?(root.getScreenCTM().a||1):1;
    root.querySelectorAll(sel).forEach((el,i)=>{
      const o=old[i];if(!o||el.hasAttribute('transform'))return;
      const n=centre(el),dx=(o[0]-n[0])/k,dy=(o[1]-n[1])/k;
      if(Math.abs(dx)+Math.abs(dy)<0.5)return;
      el.animate([{transform:`translate(${dx}px, ${dy}px)`},{transform:'translate(0px, 0px)'}],{duration:ms,easing:Motion.EASE.inOut});
    });
  }
  /* new futures and ellipse fade in; the old copies fade out as a ghost on top (opacity only) */
  function crossfade(){
    if(!mb)return;
    mb.querySelectorAll('.sc-fut, .sc-ell-mb').forEach(n=>Motion.fadeIn(n,FADE_IN_MS));
    if(!snap.ghost.length)return;
    const g=document.createElementNS(SVGNS,'g');
    g.setAttribute('class','sc-ghost');
    if(snap.ghostT)g.setAttribute('transform',snap.ghostT);
    snap.ghost.forEach(n=>g.appendChild(n));
    snap.ghost=[];
    mb.appendChild(g);
    const a=Motion.fadeOut(g,FADE_OUT_MS);
    if(a)a.onfinish=()=>g.remove();
  }

  /* The app toggles sheet.hidden on every render. Hold the sheet visible for its exit,
     and let the next render either finish the close or reopen it from mid-fade. */
  let sheetWas=sheet?!sheet.hidden:false, closing=null;
  function sheetMotion(){
    if(!sheet)return;
    let from=0;
    if(closing){
      from=+getComputedStyle(sheet).opacity;
      closing.cancel();closing=null;sheet.inert=false;sheetWas=false;
      if(sheet.hidden)return;
    }
    const open=!sheet.hidden;
    if(open&&!sheetWas){
      sheet.getAnimations().forEach(a=>a.cancel());
      sheet.animate([{opacity:from},{opacity:1}],{duration:FADE_IN_MS,easing:Motion.EASE.out});
    }else if(!open&&sheetWas){
      sheet.hidden=false;sheet.inert=true;
      closing=sheet.animate([{opacity:1},{opacity:0}],{duration:SHEET_OUT_MS,easing:Motion.EASE.out});
      closing.onfinish=()=>{sheet.hidden=true;sheet.inert=false;closing=null;};
    }
    sheetWas=open;
  }

  return function onState(state){
    const step=state&&typeof state.step==='number'?state.step:+range.value;
    const now=performance.now(), d=prev==null?0:Math.abs(step-prev), fromRange=rangeHit;
    prev=step; rangeHit=false;
    if(now-pressAt<WINDOW_MS||(fromRange&&d>1))flowUntil=now+WINDOW_MS;
    const flow=now<flowUntil&&!!snap&&d>0;
    if(flow){
      if(DOT_FLOW&&!Motion.reduced)slide(noise,'.sc-dot',snap.dots,FLOW_MS);
      if(!Motion.reduced){slide(mb,'.sc-mb-dot',snap.mbDot,FLOW_MS);slide(mb,'.sc-mb-ring',snap.mbRing,FLOW_MS);}
      crossfade();
    }
    REVEALS.forEach(el=>{const t=el.textContent.trim(),before=was.get(el)||'';
      if(t&&t!==before&&(!before||flow))Motion.fadeIn(el,FADE_IN_MS);
      was.set(el,t);});
    sheetMotion();
  };
});
