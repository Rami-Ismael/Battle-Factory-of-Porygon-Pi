/* Finite illustrative distribution; not a simulation of the DiBO implementation. */
(() => {
  const root = document.querySelector('[data-posterior-demo]');
  if (!root) return;
  const prior = [0.60, 0.35, 0.05];
  const mean = [0.60, 0.70, 0.50];
  const uncertainty = [0.05, 0.10, 0.50];
  const gamma = root.querySelector('[name=gamma]');
  const beta = root.querySelector('[name=beta]');
  const rows = [...root.querySelectorAll('[data-row]')];
  function update() {
    const g = Number(gamma.value), b = Number(beta.value);
    root.querySelector('[data-gamma]').textContent = g.toFixed(1);
    root.querySelector('[data-beta]').textContent = b.toFixed(0);
    const scores = mean.map((m, i) => m + g * uncertainty[i]);
    const logs = scores.map((s, i) => Math.log(prior[i]) + b * s);
    const max = Math.max(...logs);
    const weights = logs.map(l => Math.exp(l - max));
    const total = weights.reduce((a, v) => a + v, 0);
    rows.forEach((row, i) => {
      row.querySelector('[data-score]').textContent = scores[i].toFixed(2);
      row.querySelector('[data-prob]').textContent = (100 * weights[i] / total).toFixed(1) + '%';
      row.querySelector('progress').value = weights[i] / total;
    });
    root.querySelector('[data-result]').textContent = `Largest UCB: ${'ABC'[scores.indexOf(Math.max(...scores))]}. Largest candidate probability: ${'ABC'[weights.indexOf(Math.max(...weights))]}.`;
  }
  gamma.addEventListener('input', update);
  beta.addEventListener('input', update);
  update();
})();
