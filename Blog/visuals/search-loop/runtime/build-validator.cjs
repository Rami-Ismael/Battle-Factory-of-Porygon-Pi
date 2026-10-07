// Bundle the pinned runtime's unchanged compiled validator and data for one format.
// Usage: NODE_PATH=/path/to/ts-chacha20/node_modules node build-validator.cjs /path/to/runtime
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto');
const base=process.argv[2]||'/Users/ramiismael/.local/share/vgc-pilot-runtime/servers/diversity-baseline/8150';
const format='gen9championsvgc2026regmb';
const {TeamValidator}=require(base+'/dist/sim/team-validator');const {Teams}=require(base+'/dist/sim/teams');
TeamValidator.get(format).validateTeam(Teams.import(fs.readFileSync(path.join(__dirname,'../team-example.txt'),'utf8')));
const files=Object.keys(require.cache).filter(f=>f.startsWith(base+'/dist/')||f.includes('/ts-chacha20/'));
const names=f=>f.startsWith(base)?f.slice(base.length):'/vendor/ts-chacha20.js';
const records=files.map(f=>({id:names(f),sha256:crypto.createHash('sha256').update(fs.readFileSync(f)).digest('hex')}));
const modules=files.map(f=>JSON.stringify(names(f))+':function(module,exports,require,__dirname,__filename){\n'+fs.readFileSync(f,'utf8').replace(/^\/\/# sourceMappingURL=.*$/gm,'')+'\n}').join(',\n');
const output=`/* Pinned Pokémon Showdown validator, revision 913da3602a3aa1db79f9fdc5d5222eaf8d39569d. See validator-manifest.json and licenses. */
(function(global){'use strict';
const modules={${modules}}, cache=Object.create(null);
function normalize(s){const out=[];for(const part of s.split('/')){if(part==='..')out.pop();else if(part&&part!=='.')out.push(part)}return '/'+out.join('/')}
const paths={resolve(...parts){let acc='';for(const p of parts)acc=p.startsWith('/')?p:acc+'/'+p;return normalize(acc)},join(...parts){return normalize(parts.join('/'))},dirname(s){return s.slice(0,s.lastIndexOf('/'))||'/'}};
function deepEqual(a,b){if(Object.is(a,b))return true;if(!a||!b||typeof a!=='object'||typeof b!=='object'||Object.getPrototypeOf(a)!==Object.getPrototypeOf(b))return false;const ak=Reflect.ownKeys(a),bk=Reflect.ownKeys(b);return ak.length===bk.length&&ak.every(k=>Object.prototype.hasOwnProperty.call(b,k)&&deepEqual(a[k],b[k]))}\nfunction load(id,parent='/'){if(id==='node:util')return {isDeepStrictEqual:deepEqual};if(id==='path'||id==='node:path')return paths;if(id==='fs'||id==='node:fs')return {readdirSync(p){if(p==='/dist/data/mods')return ${JSON.stringify(fs.readdirSync(base+'/dist/data/mods'))};throw new Error('Filesystem access unavailable: '+p)}};
if(id==='ts-chacha20')id='/vendor/ts-chacha20.js';else if(id.startsWith('.'))id=paths.resolve(paths.dirname(parent),id);else if(id.startsWith('/'))id=normalize(id);
if(!modules[id]&&modules[id+'.js'])id+='.js';if(cache[id])return cache[id].exports;
if(!modules[id]){const e=new Error('Module not bundled for this format: '+id);e.code='MODULE_NOT_FOUND';throw e}
const module={exports:{}};cache[id]=module;try{modules[id](module,module.exports,x=>load(x,id),paths.dirname(id),id)}catch(e){delete cache[id];throw e}return module.exports}
const {TeamValidator}=load('/dist/sim/team-validator.js'),{Teams}=load('/dist/sim/teams.js');
const validator=TeamValidator.get('${format}');
function validateTeam(paste){try{const team=Teams.import(paste);if(!team)return ['Could not parse team'];return validator.validateTeam(team)||[]}catch(e){return ['Validator error: '+String(e.message||e)]}}
global.PokemonTeamValidator=Object.freeze({format:'${format}',revision:'913da3602a3aa1db79f9fdc5d5222eaf8d39569d',validateTeam});
})(globalThis);\n`;
fs.writeFileSync(path.join(__dirname,'showdown-validator.js'),output);
fs.writeFileSync(path.join(__dirname,'validator-manifest.json'),JSON.stringify({format,revision:'913da3602a3aa1db79f9fdc5d5222eaf8d39569d',moduleCount:files.length,bundleBytes:Buffer.byteLength(output),bundleSha256:crypto.createHash('sha256').update(output).digest('hex'),modules:records},null,2)+'\n');
fs.copyFileSync(base+'/LICENSE',path.join(__dirname,'SHOWDOWN-LICENSE'));
const vendor=files.find(f=>f.includes('/ts-chacha20/'));const vendBase=vendor.slice(0,vendor.indexOf('/build/'));
for(const license of ['LICENSE','LICENSE.md','LICENSE.txt'])if(fs.existsSync(vendBase+'/'+license)){fs.copyFileSync(vendBase+'/'+license,path.join(__dirname,'TS-CHACHA20-LICENSE'));break}
console.log(files.length,Buffer.byteLength(output));
