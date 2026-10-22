(() => {
  "use strict";

  const scoresByCase = {
    improve: [0.58, 0.72, 0.64, 0.81],
    flat: [0.41, 0.55, 0.63, 0.69],
  };
  const namesByCase = {
    improve: ["A", "B", "C", "D"],
    flat: ["E", "F", "G", "H"],
  };
  const candidateX = [560, 642, 724, 806];
  const oldBest = 0.7;
  const target = 0.78;
  const waitingY = 340;
  const scoreY = (value) => 282 - value * 240;
  const formatScore = (value) => value.toFixed(2);

  function initialize() {
    const root = document.querySelector("[data-explorer]");
    if (!root) return;

    const points = candidateX.map((_, index) => root.querySelector(`[data-new-point="${index}"]`));
    const candidates = root.querySelector("[data-candidates]");
    const targetLine = root.querySelector("[data-target-line]");
    const stepButtons = Array.from(root.querySelectorAll("[data-explorer-step]"));
    const caseButtons = Array.from(root.querySelectorAll("[data-example-case]"));
    const previous = root.querySelector("[data-explorer-prev]");
    const next = root.querySelector("[data-explorer-next]");
    const reset = root.querySelector("[data-explorer-reset]");
    const title = root.querySelector("[data-explorer-title]");
    const copy = root.querySelector("[data-explorer-copy]");
    const note = root.querySelector("[data-explorer-note]");
    const index = root.querySelector("[data-explorer-index]");
    const bestScore = root.querySelector("[data-best-score]");
    const targetScore = root.querySelector("[data-target-score]");
    const datasetSummary = root.querySelector("[data-dataset-summary]");
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    const animations = new Map();
    let stage = 0;
    let exampleCase = "improve";

    function setText(element, value) {
      if (element) element.textContent = value;
    }

    function setVisible(element, visible) {
      if (!element) return;
      element.style.visibility = visible ? "visible" : "hidden";
      element.setAttribute("aria-hidden", String(!visible));
    }

    function cancelAnimations() {
      for (const animation of animations.values()) animation.cancel();
      animations.clear();
    }

    function stageText(activeStage, values) {
      const names = namesByCase[exampleCase];
      const batchLabel = `${names[0]}–${names[3]}`;
      switch (activeStage) {
        case 0:
          return {
            title: "Fit to measured pairs",
            copy: "Six measured candidate–score pairs form the observed dataset. The best observed objective value is 0.70. Diffusion-BBO fits its inverse model to observed pairs like these.",
            note: "All values in this figure are constructed objective scores. No diffusion model is trained here, and these scores are not battle win rates.",
          };
        case 1:
          return {
            title: "Choose a request: y = 0.78",
            copy: "The dashed line marks the requested conditioning score. Diffusion-BBO uses its acquisition rule to choose a target from observed data; this example holds the request at 0.78.",
            note: "A request is a model input, not an achieved objective value. The figure does not compute the acquisition rule.",
          };
        case 2:
          return {
            title: "Generate four unmeasured candidates",
            copy: `${batchLabel} stand in for four generated candidates. Their measured objective values are not available yet, so they wait beneath the score axis rather than appearing on the requested line.`,
            note: "Candidate identities persist as you step through one batch. The two cases use different hypothetical candidates; neither is live sampling or a trained generator’s output.",
          };
        case 3:
          return {
            title: "Query the objective. Reveal the evidence.",
            copy: exampleCase === "improve"
              ? "The constructed measurements are now visible. D reaches 0.81 and raises the best observed value; A, B, and C all fall below the request of 0.78."
              : "The constructed measurements are now visible. Every candidate falls below the request of 0.78, and none improves on the existing best value of 0.70.",
            note: `Measured ${batchLabel}: ${values.map(formatScore).join(", ")}. These are illustrative labels, not observed model performance. New evidence enters through objective evaluation.`,
          };
        default:
          return {
            title: "Append measured labels. Refit. Repeat.",
            copy: "All four measured labels are appended unchanged, including outcomes below the request. A real next iteration would refit on the enlarged dataset and choose another conditioning target.",
            note: "The dataset now contains ten pairs. The requested 0.78 never replaces the measured scores. This figure represents the update; it does not perform a model refit.",
          };
      }
    }

    function render(nextStage, event, allowMotion = true) {
      if (!Number.isInteger(nextStage) || nextStage < 0 || nextStage > 4) return;

      // Read current visual positions before cancellation so rapid retargeting
      // continues from the in-flight position, not from an earlier endpoint.
      const currentTransforms = points.map((point) => point ? getComputedStyle(point).transform : "none");
      const wereCandidatesVisible = stage >= 2;
      const willShowCandidates = nextStage >= 2;
      const shouldAnimate = allowMotion && Boolean(event && event.detail > 0)
        && !reducedMotion.matches
        && wereCandidatesVisible
        && willShowCandidates;
      cancelAnimations();

      stage = nextStage;
      const measured = stage >= 3;
      const values = scoresByCase[exampleCase];
      const names = namesByCase[exampleCase];
      root.dataset.stage = String(stage);
      root.dataset.exampleCase = exampleCase;
      root.dataset.datasetPairs = String(stage >= 4 ? 10 : 6);
      setVisible(targetLine, stage >= 1);
      setVisible(candidates, willShowCandidates);

      for (const button of stepButtons) {
        button.setAttribute("aria-pressed", String(Number(button.dataset.explorerStep) === stage));
      }
      for (const button of caseButtons) {
        button.setAttribute("aria-pressed", String(button.dataset.exampleCase === exampleCase));
      }
      if (previous) previous.disabled = stage === 0;
      if (next) next.disabled = stage === 4;

      points.forEach((point, pointIndex) => {
        if (!point) return;
        const value = values[pointIndex];
        const y = measured ? scoreY(value) : waitingY;
        const destination = `translate(${candidateX[pointIndex]}px, ${y}px)`;
        const start = currentTransforms[pointIndex];

        point.style.transformOrigin = "0 0";
        point.style.transformBox = "view-box";
        point.style.transform = destination;
        point.dataset.measured = String(measured);
        point.setAttribute("aria-label", measured
          ? `Candidate ${names[pointIndex]}, illustrative measured objective ${formatScore(value)}`
          : `Candidate ${names[pointIndex]}, not yet evaluated`);
        const mark = point.querySelector("[data-point-mark]") || point.querySelector("circle, rect");
        if (mark) mark.style.fill = measured ? "currentColor" : "none";
        setText(point.querySelector("[data-point-score]"), measured ? formatScore(value) : "");
        setText(point.querySelector("[data-point-name]"), names[pointIndex]);
        for (const label of root.querySelectorAll(`[data-candidate-name="${pointIndex}"]`)) {
          label.textContent = names[pointIndex];
        }

        if (shouldAnimate && typeof point.animate === "function") {
          const destinationMatrix = getComputedStyle(point).transform;
          if (start !== destinationMatrix) {
            const animation = point.animate(
              [{ transform: start }, { transform: destination }],
              { duration: 240, easing: "cubic-bezier(.77, 0, .175, 1)" },
            );
            animations.set(point, animation);
            animation.addEventListener("finish", () => {
              if (animations.get(point) === animation) animations.delete(point);
            }, { once: true });
          }
        }
      });

      const text = stageText(stage, values);
      setText(title, text.title);
      setText(copy, text.copy);
      setText(note, text.note);
      setText(index, `${stage + 1} of 5`);
      setText(targetScore, stage >= 1 ? formatScore(target) : "—");
      setText(bestScore, formatScore(measured ? Math.max(oldBest, ...values) : oldBest));
      setText(datasetSummary, stage >= 4
        ? "10 measured pairs. All four new labels are retained."
        : measured
          ? "6 fitted pairs; 4 new measurements are ready to append."
          : "6 measured pairs in the observed dataset.");
    }

    for (const button of stepButtons) {
      button.addEventListener("click", (event) => render(Number(button.dataset.explorerStep), event));
    }
    for (const button of caseButtons) {
      button.addEventListener("click", (event) => {
        if (!Object.prototype.hasOwnProperty.call(scoresByCase, button.dataset.exampleCase)) return;
        const sameCase = exampleCase === button.dataset.exampleCase;
        exampleCase = button.dataset.exampleCase;
        render(3, event, sameCase);
      });
    }
    if (previous) previous.addEventListener("click", (event) => render(stage - 1, event));
    if (next) next.addEventListener("click", (event) => render(stage + 1, event));
    if (reset) {
      reset.addEventListener("click", (event) => {
        exampleCase = "improve";
        render(0, event);
      });
    }

    const updateMotionPreference = () => {
      if (reducedMotion.matches) cancelAnimations();
    };
    if (typeof reducedMotion.addEventListener === "function") {
      reducedMotion.addEventListener("change", updateMotionPreference);
    } else if (typeof reducedMotion.addListener === "function") {
      reducedMotion.addListener(updateMotionPreference);
    }

    render(0);
    root.dispatchEvent(new CustomEvent("explorer:ready", { bubbles: true }));
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialize, { once: true });
  } else {
    initialize();
  }
})();
