/* Runtime: builds ctx (data, sprites, plane mapping, helpers) and calls window.figInit(ctx). */
(function(){
 const state='default';
 const root=document.getElementById('fig-root');
 const show=()=>{};
 if(!window.FIGDATA||!window.SPRITES){
  root.innerHTML='<p class="fig-msg" role="alert">The team data did not load: data/figdata.js or data/sprites.js is missing. Rebuild the page with build.py, which inlines both files.</p>';return}
 const D=window.FIGDATA,S=window.SPRITES;
 const NAMES=['Trick Room','Weather','Tailwind','Other'],VARS=['--tr','--weather','--tailwind','--other'];
 const GLOSS=['a move that makes slower Pokémon move first','an ability that sets rain, sun, sand or snow','a move that doubles your side’s Speed','none of these'];
 const FIELD=['species','ability','item','move','move','move','move','nature'];const FI={species:0,ability:1,item:2,m0:3,m1:4,m2:5,m3:6,nature:7};
 const pretty=k=>{for(const t of D.teams)for(let i=0;i<6;i++)if(t.speciesId[i]===k)return t.fields[i*8];const m=D.mb607;for(let i=0;i<6;i++)if(m.speciesId[i]===k)return m.fields[i*8];return k};
 const ctx={D,S,state,show,NAMES,GLOSS,
  color:i=>`var(${VARS[i]})`,
  sprite:(key,size=48,alt)=>`<img class="sprite" src="${S[key]||''}" width="${size}" height="${size}" alt="${(alt||pretty(key)).replace(/"/g,'&quot;')}">`,
  six:(keys,size=40)=>keys.map(k=>ctx.sprite(k,size)).join(''),
  /* plane: data x,y in [-1.3,1.3] -> SVG viewBox 0..520 (y up) */
  X:x=>260+x*200,Y:y=>260-y*200,VIEW:'0 0 520 520',
  /* 1-standard-deviation ellipse as SVG attributes from varX,varY,covXY */
  ellipse(x,y,vx,vy,c,k=1){const t=(vx+vy)/2,d=Math.sqrt(Math.max(((vx-vy)/2)**2+c*c,0));const l1=Math.max(t+d,0),l2=Math.max(t-d,0);
   const ang=Math.atan2(l1-vx,c||1e-9)*180/Math.PI;return {cx:ctx.X(x),cy:ctx.Y(y),rx:Math.sqrt(l1)*200*k,ry:Math.sqrt(l2)*200*k,rotate:-(c===0?(vx>=vy?0:90):ang)}},
  /* one fixed standard-normal draw per team pushed through its 2x2 covariance (Cholesky) */
  noise:(()=>{let s=20261014;const r=()=>{s=(s*1103515245+12345)%2147483648;return s/2147483648};const g=()=>Math.sqrt(-2*Math.log(r()+1e-12))*Math.cos(2*Math.PI*r());const Z=D.teams.map(()=>[g(),g()]);
   return (i,row)=>{const [x,y,vx,vy,c]=row;const l11=Math.sqrt(Math.max(vx,0)),l21=l11>1e-9?c/l11:0,l22=Math.sqrt(Math.max(vy-l21*l21,0));return [x+l11*Z[i][0],y+l21*Z[i][0]+l22*Z[i][1]]}})(),
  /* counts at step s for a step row: [tr,weather,tailwind,other], total */
  counts:row=>row.slice(5,9),
  majority:row=>{const c=row.slice(5,9);return c.indexOf(Math.max(...c))},
  /* field index 0..47 revealed at step s (1..48) */
  fieldIndex:s=>{const o=D.order[s-1];return o.slot*8+FI[o.field]},
  /* what step s reveals for a team: {step, slot, kind, pokemon, value} */
  reveal:(team,s)=>{const o=D.order[s-1],j=FI[o.field];return {step:s,slot:o.slot,kind:FIELD[j],pokemon:team.fields[o.slot*8],value:team.fields[o.slot*8+j]}},
  /* set of field indexes revealed by step s, to mask a team sheet */
  revealedSet:s=>{const out=new Set();for(let k=1;k<=s;k++)out.add(ctx.fieldIndex(k));return out},
  futures:s=>s>=1&&s<=41?D.mb607.futures[String(s)]:(s===0?null:[]),
  FIELD};
 if(typeof window.figInit!=='function'){root.innerHTML='<p class="fig-msg" role="alert">figInit is not defined: parts/app.js did not load.</p>';return}
 try{window.figInit(ctx)}catch(e){root.insertAdjacentHTML('afterbegin','<p class="fig-msg" role="alert">Figure script error: '+String(e.message).replace(/</g,'&lt;')+'</p>');console.error(e)}
})();
