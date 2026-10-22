(() => {
  "use strict";

  const phases = [
    {
      title: "Start with measured observations",
      copy: "The dataset contains candidate inputs paired with their measured objective values. These observed pairs are the evidence used to fit the inverse model.",
      note: "This is a hypothetical trace, not a recorded experiment or a claimed battle win rate.",
    },
    {
      title: "Choose a requested score",
      copy: "Use the best observed value to propose conditioning targets, then choose a requested score y with the acquisition rule. The target guides the next generation step.",
      note: "A requested score is a model input. It is not a measured achievement.",
    },
    {
      title: "Generate candidates conditioned on y",
      copy: "Sample candidate inputs from the inverse model conditioned on the requested score. These candidates are proposals that still need evaluation.",
      note: "Conditioning on a high score does not guarantee that a candidate will achieve it.",
    },
    {
      title: "Evaluate with the true objective",
      copy: "Query the oracle for each generated candidate. In this hypothetical trace, one candidate receives a measured score below the requested target.",
      note: "New measured evidence enters here. A Pokémon application would need an evaluation protocol for estimating its battle objective.",
    },
    {
      title: "Append observations, refit, repeat",
      copy: "Add the evaluated candidate–score pairs to the dataset, refit the inverse model, and use the enlarged dataset when choosing the next conditioning target.",
      note: "Store the measured result as the new label. Do not replace it with the requested score.",
    },
  ];

  function initializeLoop() {
    const loop = document.querySelector("[data-loop]");
    if (!loop) return;

    const buttons = Array.from(loop.querySelectorAll("[data-step]"));
    const phaseMarks = Array.from(loop.querySelectorAll("[data-phase]"));
    const previous = loop.querySelector("[data-loop-prev]");
    const next = loop.querySelector("[data-loop-next]");
    const reset = loop.querySelector("[data-loop-reset]");
    const title = loop.querySelector("[data-loop-title]");
    const copy = loop.querySelector("[data-loop-copy]");
    const note = loop.querySelector("[data-loop-note]");
    const index = loop.querySelector("[data-loop-index]");
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    const activeAnimations = new Map();
    let activeStep = 0;

    function cancelAnimations() {
      for (const animation of activeAnimations.values()) animation.cancel();
      activeAnimations.clear();
    }

    function showStep(requestedStep, event) {
      const step = Math.max(0, Math.min(phases.length - 1, requestedStep));
      if (!Number.isInteger(step)) return;

      const incoming = phaseMarks.find((mark) => Number(mark.dataset.phase) === step);
      const fromOpacity = incoming ? getComputedStyle(incoming).opacity : "1";
      const isPointerChange = Boolean(event && event.detail > 0);
      const shouldAnimate = step !== activeStep && isPointerChange && !reducedMotion.matches;
      cancelAnimations();

      activeStep = step;
      loop.dataset.activeStep = String(step);
      for (const button of buttons) {
        button.setAttribute("aria-pressed", String(Number(button.dataset.step) === step));
      }
      for (const mark of phaseMarks) {
        mark.dataset.active = String(Number(mark.dataset.phase) === step);
      }

      const phase = phases[step];
      if (title) title.textContent = phase.title;
      if (copy) copy.textContent = phase.copy;
      if (note) note.textContent = phase.note;
      if (index) index.textContent = `${step + 1} of ${phases.length}`;
      if (previous) previous.disabled = step === 0;
      if (next) next.disabled = step === phases.length - 1;

      if (shouldAnimate && incoming && typeof incoming.animate === "function") {
        const toOpacity = getComputedStyle(incoming).opacity;
        const animation = incoming.animate(
          [{ opacity: fromOpacity }, { opacity: toOpacity }],
          { duration: 180, easing: "cubic-bezier(.23, 1, .32, 1)" },
        );
        activeAnimations.set(incoming, animation);
        animation.addEventListener("finish", () => {
          if (activeAnimations.get(incoming) === animation) activeAnimations.delete(incoming);
        }, { once: true });
      }
    }

    for (const button of buttons) {
      button.addEventListener("click", (event) => showStep(Number(button.dataset.step), event));
    }
    if (previous) previous.addEventListener("click", (event) => showStep(activeStep - 1, event));
    if (next) next.addEventListener("click", (event) => showStep(activeStep + 1, event));
    if (reset) reset.addEventListener("click", (event) => showStep(0, event));

    const handleMotionPreference = () => {
      if (reducedMotion.matches) cancelAnimations();
    };
    if (typeof reducedMotion.addEventListener === "function") {
      reducedMotion.addEventListener("change", handleMotionPreference);
    } else if (typeof reducedMotion.addListener === "function") {
      reducedMotion.addListener(handleMotionPreference);
    }

    showStep(0);
  }

  function initializeReadingPlan() {
    const storageKey = "rami.reading-guide.20261020.v1";
    const taskIds = ["task-1", "task-2", "task-3", "task-4"];
    const tasks = taskIds
      .map((id) => document.getElementById(id))
      .filter((task) => task && task.matches('input[type="checkbox"][data-task]'));
    if (!tasks.length) return;

    const completedMinutes = document.querySelector("[data-completed-minutes]");
    const completedCount = document.querySelector("[data-completed-count]");
    const progressFill = document.querySelector("[data-progress-fill]");
    const progressTrack = document.querySelector("[data-progress-track]");
    const saveStatus = document.querySelector("[data-save-status]");
    const reset = document.querySelector("[data-task-reset]");
    const totalMinutes = 60;
    let storage = null;

    try {
      storage = window.localStorage;
      const saved = storage.getItem(storageKey);
      if (saved !== null) {
        try {
          const completedIds = JSON.parse(saved);
          if (Array.isArray(completedIds) && completedIds.every((id) => taskIds.includes(id))) {
            for (const task of tasks) task.checked = completedIds.includes(task.id);
          }
        } catch {
          // An invalid saved value leaves the initial checklist intact.
        }
      }
    } catch {
      storage = null;
    }

    function saveProgress() {
      let saved = false;
      if (storage) {
        try {
          storage.setItem(storageKey, JSON.stringify(tasks.filter((task) => task.checked).map((task) => task.id)));
          saved = true;
        } catch {
          storage = null;
        }
      }
      if (saveStatus) {
        saveStatus.textContent = saved ? "Progress saved in this browser" : "Progress lasts for this visit";
      }
    }

    function updateProgress() {
      const checked = tasks.filter((task) => task.checked);
      const minutes = checked.reduce((sum, task) => {
        const taskMinutes = Number(task.dataset.minutes);
        return sum + (Number.isFinite(taskMinutes) && taskMinutes > 0 ? taskMinutes : 0);
      }, 0);
      const fraction = Math.max(0, Math.min(1, minutes / totalMinutes));
      if (completedMinutes) completedMinutes.textContent = String(minutes);
      if (completedCount) completedCount.textContent = String(checked.length);
      if (progressFill) progressFill.style.transform = `scaleX(${fraction})`;
      if (progressTrack) {
        progressTrack.setAttribute("role", "progressbar");
        progressTrack.setAttribute("aria-valuemin", "0");
        progressTrack.setAttribute("aria-valuemax", String(totalMinutes));
        progressTrack.setAttribute("aria-valuenow", String(Math.min(totalMinutes, minutes)));
        progressTrack.setAttribute("aria-valuetext", `${minutes} of ${totalMinutes} planned minutes complete`);
      }
      saveProgress();
    }

    for (const task of tasks) task.addEventListener("change", updateProgress);
    if (reset) {
      reset.addEventListener("click", () => {
        for (const task of tasks) task.checked = false;
        updateProgress();
      });
    }
    updateProgress();
  }

  function initialize() {
    initializeLoop();
    initializeReadingPlan();
    for (const element of document.querySelectorAll("[data-js-only]")) element.hidden = false;
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialize, { once: true });
  } else {
    initialize();
  }
})();
