// Extract a reproducible counting domain from the existing frozen browser validator.
// The source bundle is never modified. All output is local to this figure.
const fs = require('fs'), path = require('path'), vm = require('vm'), crypto = require('crypto');
const out = __dirname;
const bundle = path.resolve(out, '../search-loop/runtime/showdown-validator.js');
const bytes = fs.readFileSync(bundle);
const source = bytes.toString().replace('global.PokemonTeamValidator=Object.freeze',
  'global.countingInternals={validator,Teams};global.PokemonTeamValidator=Object.freeze');
vm.runInThisContext(source);
const {validator: v} = global.countingInternals, d = v.dex;
const stats = {hp:32,atk:32,def:2,spa:0,spd:0,spe:0};
const setFor = (s, ability, moves, item='') => ({species:s.name,ability,moves,item,nature:'Hardy',level:50,evs:{...stats}});
const failures = [], raw = [];
let checks = 0;
for (const s of d.species.all().filter(s => !s.isNonstandard && !s.battleOnly && !s.isMega)) {
  // Conservative subset: only explicit Champions 9M sources, not inherited/event moves.
  const directLearnset=d.data.Learnsets[s.id]?.learnset || {};
  const allMoves = Object.keys(directLearnset).filter(m => directLearnset[m].includes('9M') && !d.moves.get(m).isNonstandard).sort();
  const abilities = [...new Set(Object.values(s.abilities))];
  const pairs = [];
  for (const ability of abilities) {
    const moves = [];
    for (const move of allMoves) {
      const errors = v.validateSet(setFor(s,ability,[move])); checks++;
      if (!errors) moves.push(move); else failures.push({species:s.id,ability,move,errors});
    }
    if(moves.length) pairs.push({ability,moves});
  }
  if (!pairs.length) continue;
  raw.push({id:s.id,name:s.name,num:s.num,baseSpecies:s.baseSpecies,types:s.types,baseStats:s.baseStats,
    weightkg:s.weightkg,gender:s.gender||null,genderRatio:s.genderRatio||null,pairs,
    sourceKinds:[...new Set(allMoves.flatMap(m=>directLearnset[m]))]});
}
// Merge only same-Dex forms with the same modeled battle properties. Appearance is excluded.
const groups = new Map();
for(const r of raw) {
  const signature=JSON.stringify([r.num,r.types,r.baseStats,r.weightkg,r.gender,r.genderRatio,r.pairs]);
  if(!groups.has(signature)) groups.set(signature,{...r,aliases:[]});
  groups.get(signature).aliases.push(r.name);
}
const roster=[...groups.values()].sort((a,b)=>a.num-b.num||a.id.localeCompare(b.id));
const items=d.items.all().filter(i=>!i.isNonstandard).map(i=>({id:i.id,name:i.name,megaStone:i.megaStone||null}));
const itemErrors=[];
for(const r of roster) for(const it of [{id:'',name:'None'},...items]) {
  const p=r.pairs[0],err=v.validateSet(setFor(r,p.ability,p.moves.slice(0,4),it.id)); checks++;
  if(err) itemErrors.push({species:r.id,item:it.id,errors:err});
}
const alignments=d.natures.all().filter(n=>n.plus||n.id==='hardy').map(n=>({id:n.id,name:n.name,plus:n.plus||null,minus:n.minus||null}));
// Fixed-seed combinations test full team legality at all construction layers.
let state=9182026; const rand=()=>((state=(Math.imul(1664525,state)+1013904223)>>>0)/4294967296);
const choose=a=>a[Math.floor(rand()*a.length)];
const shuffle=a=>{a=[...a];for(let i=a.length-1;i>0;i--){let j=Math.floor(rand()*(i+1));[a[i],a[j]]=[a[j],a[i]];}return a;};
const teamErrors=[];
for(let t=0;t<500;t++) {
  const selected=[],seen=new Set();
  while(selected.length<6){const r=choose(roster);if(!seen.has(r.num)){seen.add(r.num);selected.push(r);}}
  const its=shuffle(items).slice(0,6);
  const team=selected.map((r,i)=>{const p=choose(r.pairs),set=setFor(r,p.ability,shuffle(p.moves).slice(0,Math.min(4,p.moves.length)),its[i].id);
    set.nature=choose(alignments).name;set.evs={hp:0,atk:0,def:0,spa:0,spd:0,spe:0};
    const keys=Object.keys(set.evs); for(let k=0;k<66;k++){const avail=keys.filter(s=>set.evs[s]<32);set.evs[choose(avail)]++;}return set;});
  const errors=v.validateTeam(team);checks++; if(errors)teamErrors.push({team,errors});
}
const output={format:global.PokemonTeamValidator.format,revision:global.PokemonTeamValidator.revision,
  bundleSha256:crypto.createHash('sha256').update(bytes).digest('hex'),created:'2026-09-18',
  rules:{teamSize:6,statTotal:v.ruleTable.evLimit,statCap:32,moveCount:'exactly four, or all available if fewer',moveSource:'direct 9M only',allowNoItem:true},
  rawFormCount:raw.length,roster,items,alignments,
  validation:{checks,singleMoveFailures:failures,itemErrors,teamErrors,randomTeams:500}};
fs.writeFileSync(path.join(out,'domain.json'),JSON.stringify(output,null,2));
console.log(JSON.stringify({forms:roster.length,species:new Set(roster.map(r=>r.num)).size,items:items.length,alignments:alignments.length,checks,moveFailures:failures.length,itemErrors:itemErrors.length,teamErrors:teamErrors.length,sourceKinds:[...new Set(roster.flatMap(r=>r.sourceKinds))],merged:roster.filter(r=>r.aliases.length>1).map(r=>({name:r.name,aliases:r.aliases.length}))},null,2));
