/* Minimal retrieval-practice quiz widget. Reusable across lessons.
   Markup:
   <div class="quiz" data-answer="a">
     <p class="q">Question text?</p>
     <button data-opt="a">…</button> <button data-opt="b">…</button> …
     <div class="why">Explanation shown after any answer.</div>
   </div>
   Immediate feedback; the learner may retry a wrong answer. Options should
   have equal word counts so formatting never leaks the answer. */
(function () {
  const css = `
  .quiz { border: 1px solid var(--rule,#ccc); border-radius: 6px; padding: 1rem 1.2rem; margin: 1.4rem 0; max-width: 42rem; }
  .quiz .q { margin: 0 0 .7rem; font-weight: 600; }
  .quiz .opts { display: grid; gap: .45rem; }
  .quiz button { font: inherit; text-align: left; padding: .5rem .8rem; border: 1px solid var(--rule,#ccc); border-radius: 4px; background: transparent; color: inherit; cursor: pointer; }
  .quiz button:hover { border-color: var(--accent,#8a3b12); }
  .quiz button.right { background: var(--ok-bg,#e5f1e9); border-color: var(--cover,#2c6e49); }
  .quiz button.wrong { background: var(--bad-bg,#f5e3dc); border-color: var(--warn,#9c4a1a); }
  .quiz .why { display: none; margin-top: .8rem; font-size: .9em; color: var(--ink-soft,#555); }
  .quiz.done .why { display: block; }
  .quiz .score { font-size: .8em; color: var(--ink-soft,#555); margin-top: .5rem; }
  .quiz-summary { margin: 1.5rem 0; font-style: italic; color: var(--ink-soft,#555); }`;
  const style = document.createElement('style'); style.textContent = css; document.head.appendChild(style);

  const quizzes = Array.from(document.querySelectorAll('.quiz'));
  const state = { first: {}, total: quizzes.length };
  quizzes.forEach((qz, i) => {
    const answer = qz.dataset.answer;
    const buttons = Array.from(qz.querySelectorAll('button[data-opt]'));
    // wrap options in a grid container if not already
    if (!qz.querySelector('.opts')) {
      const wrap = document.createElement('div'); wrap.className = 'opts';
      buttons[0].parentNode.insertBefore(wrap, buttons[0]);
      buttons.forEach(b => wrap.appendChild(b));
    }
    const score = document.createElement('div'); score.className = 'score'; qz.appendChild(score);
    buttons.forEach(b => b.addEventListener('click', () => {
      const ok = b.dataset.opt === answer;
      if (!(i in state.first)) state.first[i] = ok;
      buttons.forEach(x => x.classList.remove('right', 'wrong'));
      b.classList.add(ok ? 'right' : 'wrong');
      if (ok) { qz.classList.add('done'); score.textContent = state.first[i] ? 'Correct first try.' : 'Correct — note which distractor pulled you first; that is the thing to re-read.'; }
      else { score.textContent = 'Not that one — try again before reading on.'; }
      updateSummary();
    }));
  });
  function updateSummary() {
    let el = document.querySelector('.quiz-summary');
    if (!el) return;
    const answered = Object.keys(state.first).length;
    const firstRight = Object.values(state.first).filter(Boolean).length;
    el.textContent = answered ? `First-try score so far: ${firstRight} / ${answered} of ${state.total}.` : '';
  }
})();
