import htmx from 'htmx.org';
import { Editor, Mark } from '@tiptap/core';
import StarterKit from '@tiptap/starter-kit';
import Placeholder from '@tiptap/extension-placeholder';
import { Plugin } from '@tiptap/pm/state';
import { Decoration, DecorationSet } from '@tiptap/pm/view';
import { createIcons, icons } from 'lucide';

window.htmx = htmx;
const $ = (id) => document.getElementById(id);
const state = JSON.parse($('document-state').textContent);
let comments = state.comments;
let version = state.version;
let generation = 0;
let savedGeneration = 0;
let pendingSave = null;
let saveTimer;
let activeId = comments.find(n => n.status === 'open')?.id;
let selectedRange;
let previewEditor;
let previewRevision;
let blockIndex = 0;
let blockPosition = null;
const escape = text => String(text).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const icon = name => `<i data-lucide="${name}" aria-hidden="true"></i>`;
const drawIcons = () => createIcons({ icons });
const Annotation = Mark.create({
  name: 'annotation',
  inclusive: false,
  addAttributes() { return { id: {default:null, parseHTML: el => el.getAttribute('data-annotation'), renderHTML: attrs => ({'data-annotation': attrs.id})} }; },
  parseHTML() { return [{tag:'mark[data-annotation]'}]; },
  renderHTML({HTMLAttributes}) { return ['mark', HTMLAttributes, 0]; },
  addProseMirrorPlugins() {
    return [new Plugin({props:{decorations(editorState) {
      const decorations = [];
      editorState.doc.descendants((node,pos) => {
        const mark = node.marks.find(m => m.type.name === 'annotation');
        if (!node.isText || !mark) return;
        const classes = [];
        if (mark.attrs.id === activeId) classes.push('active-annotation');
        if (comments.find(n=>n.id===mark.attrs.id)?.status !== 'open') classes.push('resolved-annotation');
        if (classes.length) decorations.push(Decoration.inline(pos,pos+node.nodeSize,{class:classes.join(' ')}));
      });
      return DecorationSet.create(editorState.doc,decorations);
    }}})];
  },
});

const editor = new Editor({
  element: $('editor'),
  extensions: [StarterKit.configure({link:{openOnClick:false}}), Annotation, Placeholder.configure({placeholder:'Start writing. A first draft only needs a first sentence…'})],
  content: state.content,
  editorProps: {attributes:{'aria-label':'Document body',role:'textbox','aria-multiline':'true'}, handleKeyDown(view,event) {
    if (!$('block-menu').hidden) {
      if (event.key === 'Escape') {hideBlockMenu(); return true;}
      if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
        blockIndex = (blockIndex + (event.key === 'ArrowDown' ? 1 : -1) + blocks.length)%blocks.length;
        drawBlockMenu(); return true;
      }
      if (event.key === 'Enter') {insertBlock(blockIndex); return true;}
      hideBlockMenu();
    }
    if (event.key === '/' && view.state.selection.empty && view.state.selection.$from.parentOffset === 0 && view.state.selection.$from.parent.isTextblock) {
      blockPosition = view.state.selection.from;
      blockIndex = 0;
      drawBlockMenu();
      const rect = view.coordsAtPos(blockPosition);
      $('block-menu').style.left = `${Math.max(8,Math.min(rect.left,innerWidth-240))}px`;
      $('block-menu').style.top = `${Math.max(8,Math.min(rect.bottom+8,innerHeight-330))}px`;
      $('block-menu').hidden = false;
      return true;
    }
    return false;
  }},
  onUpdate() { changed(); updateStats(); refreshMarkStyles(); renderNotes(); },
  onSelectionUpdate() { updateToolbar(); },
});
// Exposed for local integration tests and future workshop extensions.
window.workshop = { editor, save: () => flush(), get comments() {return comments;} };

const blocks = [
  {name:'Text',hint:'A plain paragraph',icon:'type',command:c=>c.setParagraph()},
  {name:'Heading 1',hint:'A new chapter',icon:'heading-1',command:c=>c.setHeading({level:1})},
  {name:'Heading 2',hint:'A section within it',icon:'heading-2',command:c=>c.setHeading({level:2})},
  {name:'Bullet list',hint:'Thoughts, one at a time',icon:'list',command:c=>c.toggleBulletList()},
  {name:'Numbered list',hint:'Give your thoughts an order',icon:'list-ordered',command:c=>c.toggleOrderedList()},
  {name:'Quote',hint:'A voice within the page',icon:'quote',command:c=>c.toggleBlockquote()},
  {name:'Divider',hint:'A small pause',icon:'minus',command:c=>c.setHorizontalRule()},
];
function hideBlockMenu() {$('block-menu').hidden = true;blockPosition = null;}
function drawBlockMenu() {
  $('block-menu').innerHTML = blocks.map((block,i)=>`<button role="menuitem" data-block="${i}" class="${i===blockIndex?'chosen':''}">${icon(block.icon)}<span>${block.name}<small>${block.hint}</small></span></button>`).join('');
  drawIcons();
}
function insertBlock(index) {
  const position = blockPosition;
  hideBlockMenu();
  if (position !== null) blocks[index].command(editor.chain().focus().setTextSelection(position)).run();
}
$('block-menu').addEventListener('mousedown',event=>event.preventDefault());
$('block-menu').addEventListener('click',event=>{const button=event.target.closest('[data-block]');if(button) insertBlock(Number(button.dataset.block));});
document.addEventListener('mousedown',event=>{if (!event.target.closest('#block-menu') && !event.target.closest('#editor')) hideBlockMenu();});

function toast(message, persistent=false) {
  $('toast').textContent = message;
  $('toast').hidden = false;
  clearTimeout(toast.timer);
  if (!persistent) toast.timer = setTimeout(() => $('toast').hidden = true, 4500);
}
function changed() {
  generation++;
  $('save-status').textContent = 'Unsaved changes';
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => save(), 1000);
}
async function api(url, options={}) {
  const response = await fetch(url, { ...options, headers: {'Content-Type':'application/json', ...options.headers} });
  if (!response.ok) {
    let message = 'Could not save. Your draft is still here; check the server and try again.';
    try { message = (await response.json()).error || message; } catch {}
    throw new Error(message);
  }
  return response.status === 204 ? null : response.json();
}
async function save(options={}) {
  clearTimeout(saveTimer);
  if (pendingSave) {
    const success = await pendingSave;
    if (!success) return false;
  }
  if (generation === savedGeneration && !options.checkpoint) return true;
  const captured = generation;
  const payload = { title:$('document-title').value, content:editor.getJSON(), comments:structuredClone(comments), version, ...options };
  $('save-status').textContent = 'Saving…';
  pendingSave = (async () => {
    try {
      const data = await api(`/api/documents/${state.id}`, {method:'PUT', body:JSON.stringify(payload)});
      version = data.version;
      savedGeneration = captured;
      $('save-status').innerHTML = captured === generation ? '<span class="status-dot"></span> All changes saved' : 'Unsaved changes';
      const title = payload.title.trim() || 'Untitled document';
      $('breadcrumb-title').textContent = title;
      document.title = `${title} · Margin`;
      document.querySelector('.document-link.selected span').textContent = title;
      return true;
    } catch (error) {
      $('save-status').textContent = 'Save failed · Ctrl/⌘ S to retry';
      toast(error.message, true);
      return false;
    }
  })();
  const success = await pendingSave;
  pendingSave = null;
  return success;
}
async function flush() {
  if (!await save()) return false;
  while (generation !== savedGeneration) if (!await save()) return false;
  return true;
}
function updateStats() {
  const text = editor.getText().trim();
  const count = text ? text.split(/\s+/u).length : 0;
  $('word-count').textContent = `${count.toLocaleString()} ${count === 1 ? 'word' : 'words'}`;
  $('reading-time').textContent = `${Math.max(1, Math.ceil(count/220))} min read`;
}
function updateToolbar() {
  const mapping = {bold:'bold',italic:'italic',h2:'heading',bullet:'bulletList',quote:'blockquote',paragraph:'paragraph'};
  document.querySelectorAll('[data-format]').forEach(button => {
    const format = button.dataset.format;
    const active = mapping[format] && editor.isActive(mapping[format], format === 'h2' ? {level:2} : {});
    button.classList.toggle('active', Boolean(active));
    button.setAttribute('aria-pressed', String(Boolean(active)));
  });
  $('annotate-button').disabled = editor.state.selection.empty;
}
document.querySelectorAll('[data-format]').forEach(button => {
  button.addEventListener('mousedown', e => e.preventDefault());
  button.addEventListener('click', () => {
    const chain = editor.chain().focus();
    const actions = {paragraph:()=>chain.setParagraph(),bold:()=>chain.toggleBold(),italic:()=>chain.toggleItalic(),h2:()=>chain.toggleHeading({level:2}),bullet:()=>chain.toggleBulletList(),quote:()=>chain.toggleBlockquote(),undo:()=>chain.undo(),redo:()=>chain.redo()};
    actions[button.dataset.format]().run();
    updateToolbar();
  });
});
function resizeTitle() {
  $('document-title').style.height = 'auto';
  $('document-title').style.height = `${$('document-title').scrollHeight}px`;
}
$('document-title').addEventListener('input', () => {resizeTitle(); changed();});
window.addEventListener('resize', resizeTitle);
function rangesFor(id) {
  const ranges = [];
  editor.state.doc.descendants((node,pos) => {
    if (node.isText && node.marks.some(m => m.type.name === 'annotation' && m.attrs.id === id)) ranges.push({from:pos,to:pos+node.nodeSize});
  });
  return ranges;
}
function quoteFor(note) {
  const ranges = rangesFor(note.id);
  return ranges.length ? editor.state.doc.textBetween(ranges[0].from, ranges.at(-1).to,' ') : note.quote;
}
function filteredNotes(openOnly=false) {
  const pass = $('pass-filter').value;
  return comments.filter(note => (pass === 'all' || note.pass === pass) && (!openOnly || note.status === 'open'));
}
function refreshMarkStyles() {
  editor.view.dispatch(editor.state.tr.setMeta('annotationStyle',true));
}
function renderNotes() {
  const notes = filteredNotes();
  const open = filteredNotes(true);
  $('open-count').textContent = comments.filter(n => n.status === 'open').length;
  if (!open.some(n => n.id === activeId)) activeId = open[0]?.id;
  const index = open.findIndex(n => n.id === activeId);
  $('suggestion-position').textContent = open.length ? `${index+1} of ${open.length} suggestions` : 'No open suggestions';
  $('previous-note').disabled = open.length < 2;
  $('next-note').disabled = open.length < 2;
  $('notes-list').innerHTML = notes.length ? [...notes].sort((a,b) => (a.status !== 'open')-(b.status !== 'open')).map((note,i) => {
    const resolved = note.status !== 'open';
    const detached = !rangesFor(note.id).length;
    return `<article class="note-card ${note.id === activeId ? 'active' : ''} ${resolved ? 'resolved-card' : ''}" data-note="${escape(note.id)}" tabindex="0" aria-label="${escape(note.pass)} note">
      <div class="note-top"><span class="pass-badge">${escape(note.pass)}</span><span class="note-number">${resolved ? 'Resolved' : String(i+1).padStart(2,'0')}</span></div>
      <blockquote class="note-quote">“${escape(quoteFor(note))}”</blockquote><p class="note-body">${escape(note.body)}</p>
      ${note.replacement ? `<div class="replacement">${escape(note.replacement)}</div>` : ''}
      ${detached ? '<span class="detached-label">This passage has been removed. Your note is kept.</span>' : ''}
      <div class="note-actions"><button data-action="resolve">${icon(resolved ? 'rotate-ccw' : 'check')}${resolved ? 'Reopen' : 'Resolve'}</button>${!resolved && note.replacement && !detached ? `<button class="accept" data-action="accept">Accept wording ${icon('arrow-up-right')}</button>` : ''}</div></article>`;
  }).join('') : `<div class="empty-notes">${icon('highlighter')}<p>${$('pass-filter').value === 'all' ? 'Room for a closer look.' : 'No notes in this editing pass.'}<br>Select a passage and add your first note.</p></div>`;
  refreshMarkStyles();
  drawIcons();
}
function selectNote(id, scroll=true) {
  activeId = id;
  renderNotes();
  const ranges = rangesFor(id);
  if (scroll && ranges.length) {
    const element = [...$('editor').querySelectorAll('[data-annotation]')].find(el => el.dataset.annotation === id);
    element?.scrollIntoView({block:'center',behavior:matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth'});
  }
}
function stepNote(direction) {
  const notes = filteredNotes(true);
  if (!notes.length) return;
  const index = notes.findIndex(n => n.id === activeId);
  selectNote(notes[(index + direction + notes.length)%notes.length].id);
}
$('previous-note').onclick = () => stepNote(-1);
$('next-note').onclick = () => stepNote(1);
$('pass-filter').onchange = renderNotes;
$('editor').addEventListener('click', event => {
  const mark = event.target.closest('[data-annotation]');
  if (mark) {
    const note = comments.find(n => n.id === mark.dataset.annotation);
    if ($('pass-filter').value !== 'all' && note?.pass !== $('pass-filter').value) $('pass-filter').value = 'all';
    selectNote(mark.dataset.annotation,false);
    [...$('notes-list').children].find(el => el.dataset.note === mark.dataset.annotation)?.scrollIntoView({block:'nearest'});
  }
});
$('notes-list').addEventListener('click', event => {
  const card = event.target.closest('[data-note]');
  if (!card) return;
  const note = comments.find(n => n.id === card.dataset.note);
  const action = event.target.closest('[data-action]')?.dataset.action;
  if (action === 'resolve') {
    note.status = note.status === 'open' ? 'resolved' : 'open';
    changed(); renderNotes();
  } else if (action === 'accept') {
    const ranges = rangesFor(note.id);
    if (!ranges.length) return toast('The highlighted passage has been removed.');
    const from = ranges[0].from, to = ranges.at(-1).to;
    note.status = 'accepted';
    editor.chain().focus().setTextSelection({from,to}).unsetMark('annotation').insertContent({type:'text',text:note.replacement}).run();
    changed(); renderNotes(); toast('Suggested wording accepted. Undo with ⌘ / Ctrl + Z.');
  } else selectNote(note.id);
});
$('notes-list').addEventListener('keydown', event => {
  if (event.key === 'Enter' && event.target.matches('[data-note]')) selectNote(event.target.dataset.note);
});
$('annotate-button').addEventListener('mousedown', event => event.preventDefault());
$('annotate-button').onclick = () => {
  const {from,to,empty} = editor.state.selection;
  if (empty) return toast('Select a passage first.');
  let overlap = false;
  editor.state.doc.nodesBetween(from,to,node => { if (node.marks.some(m => m.type.name === 'annotation')) overlap = true; });
  if (overlap) return toast('This passage already has a note. Choose a passage without a highlight.');
  selectedRange = {from,to};
  $('selected-quote').textContent = editor.state.doc.textBetween(from,to,' ');
  $('note-form').reset();
  $('note-dialog').showModal();
  $('note-body').focus();
};
$('note-form').onsubmit = event => {
  event.preventDefault();
  const id = crypto.randomUUID();
  comments.push({id,quote:$('selected-quote').textContent,body:$('note-body').value.trim(),replacement:$('note-replacement').value.trim(),pass:$('note-pass').value,status:'open'});
  editor.chain().focus().setTextSelection(selectedRange).setMark('annotation',{id}).setTextSelection(selectedRange.to).run();
  activeId = id;
  $('note-dialog').close(); changed(); renderNotes();
};
document.querySelectorAll('[data-close]').forEach(button => button.onclick = () => $(button.dataset.close).close());
document.querySelectorAll('dialog').forEach(dialog => dialog.addEventListener('click', event => {
  const rect = dialog.getBoundingClientRect();
  if (event.target === dialog && (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom)) dialog.close();
}));
$('new-document').onclick = async () => { if (await flush()) {$('new-dialog').showModal(); $('new-title').focus();} };
$('help-button').onclick = () => $('help-dialog').showModal();
$('toggle-library').onclick = () => document.querySelector('.library').classList.toggle('visible');
$('checkpoint-button').onclick = () => {$('checkpoint-dialog').showModal(); $('revision-label').focus();};
$('checkpoint-form').onsubmit = async event => {
  event.preventDefault();
  const button = event.submitter;
  button.disabled = true;
  if (await flush() && await save({checkpoint:true,label:$('revision-label').value.trim(),major:$('revision-major').checked})) {
    $('checkpoint-dialog').close(); event.target.reset(); toast('Revision saved. Find it in History.');
  }
  button.disabled = false;
};
async function loadHistory() {
  await htmx.ajax('GET',`/documents/${state.id}/history`,{target:'#history-content',swap:'innerHTML'});
  drawIcons();
}
$('history-button').onclick = async () => {
  if (!await flush()) return;
  $('revision-preview').hidden = true;
  $('history-content').textContent = 'Loading revisions…';
  $('history-dialog').showModal();
  try {await loadHistory();} catch {toast('Could not load history. Try again.');}
};
$('history-content').addEventListener('click', async event => {
  const flag = event.target.closest('[data-flag]');
  const revision = event.target.closest('[data-revision]');
  try {
    if (flag) {await api(`/api/documents/${state.id}/revisions/${flag.dataset.flag}/flag`,{method:'POST'}); await loadHistory();}
    if (revision) {
      const data = await api(`/api/documents/${state.id}/revisions/${revision.dataset.revision}`);
      previewRevision = data.id;
      $('preview-title').textContent = data.title;
      previewEditor?.destroy();
      $('preview-editor').innerHTML = '';
      previewEditor = new Editor({element:$('preview-editor'),extensions:[StarterKit,Annotation],content:data.content,editable:false});
      $('revision-preview').hidden = false;
      $('revision-preview').scrollIntoView({block:'nearest'});
    }
  } catch(error) {toast(error.message);}
});
$('restore-revision').onclick = async () => {
  if (!previewRevision || !await flush()) return;
  try {await api(`/api/documents/${state.id}/revisions/${previewRevision}/restore`,{method:'POST',body:JSON.stringify({version})}); location.reload();} catch(error) {toast(error.message);}
};
document.querySelectorAll('.document-link,.brand').forEach(link => link.addEventListener('click', async event => {
  if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
  event.preventDefault(); if (await flush()) location.href = link.href;
}));
document.addEventListener('keydown', event => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 's') {event.preventDefault(); flush().then(ok => ok && toast('All changes saved.'));}
});
window.addEventListener('beforeunload', event => {if (generation !== savedGeneration) {event.preventDefault(); event.returnValue = '';}});
document.addEventListener('htmx:responseError', () => toast('The request failed. Check the local server and try again.'));
resizeTitle(); updateStats(); updateToolbar(); renderNotes(); drawIcons();
