# Story and image quality-diversity tutorials

## Scope and implementation fact sheet

### Takeaway
Retrieved 11 November 2026. Stable documentation identifies itself as v0.12.1. The release-tag notebooks, including stored outputs, are transcribed below for exact prompts, constructor arguments, algorithms, and examples. Neither notebook pins a ribs version in its pip command: version pin **not found**. The documentation release version is distinct from a dependency pin. [Story docs](https://docs.pyribs.org/en/stable/tutorials/qdaif.html), [image docs](https://docs.pyribs.org/en/stable/tutorials/qdhf.html).

### Cited Findings
- Stories: one solution is a Python str, scalar shape (), stored as object dtype. The archive is a 20×20 GridArchive, 400 cells. Llama 3.1 8B instruct q4_K_M both mutates and rates stories. Objective is short-story quality; measures are romance and happy ending. Each score clips ratings to [1,10] and averages five calls, thus 15 rating calls per candidate. Exact prompts and all six archive arguments appear in the cell transcript below. The custom LLMDirectionalEmitter subclasses EmitterBase, returns three seed stories when empty, otherwise samples an elite and independently chooses up/down for each measure before calling its structured-output mutation chain. No tell override is defined; inherited EmitterBase tell supplies no adaptation. Scheduler(archive, emitters) performs ask → evaluate → tell. Run: 200 iterations, batch 1 after the three-story initialization; metrics every 5, heatmap display every 50. Stated runtime ~1 hour on Colab T4, ~half an hour on RTX A6000; saved progress ends at 35:21. Final coverage 28.5%, raw QD score 955.8. Maximum objective is not logged: not found. Heatmap x=Romance Story, y=Happy Ending, color=quality [1,10]. [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

- Images: one solution is a flattened 4×32×32 Stable Diffusion noise latent, length 4096; evaluation reshapes batch to (4,4,32,32) and produces 256×256 images using lambdalabs/miniSD-diffusers. Prompt is "an astronaut riding a horse on mars". CLIP ViT-B/32 scores each image once; no averaging. For raw image/text logit L, code computes (10 - 100/L)*10. The comment describes remapping [0,10] minimization to [0,100] maximization, but no clipping enforces that objective range. Measures are a learned linear projection of 512-dimensional CLIP image features into two real-valued coordinates, with no fixed semantic names or bounded activation. DivProj uses one nn.Linear layer, squared L2 distances, margin 0.05, Adam lr=1e-3, batch 32, maximum 200 epochs, sequential 75%/25% train/validation split, and moving-window validation early stopping. Exact loss and label code follow below. [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

- DreamSim uses pretrained=True, dreamsim_type="open_clip_vitb32". The simulated preference is 1 if cosine(ref,x2)>cosine(ref,x1), otherwise 0. Retraining iterations [1,21,51,101] add 1000 random triplets each, retaining previous triplets. fit_div_proj creates a fresh model and optimizer each time, despite prose calling this fine-tuning. Initial population is 200 standard-normal latents. At later retrains all current archive elites are regenerated and rescored; only those current elites, not all historical candidates, are re-added to a new archive. Bounds are the 1st and 99th percentiles of newly projected coordinates over those solutions. GridArchive(solution_dim=len(sols[0]), dims=(20,20), ranges=archive_bounds, seed=123), 400 cells. New GaussianEmitter(archive=archive, sigma=0.1, initial_solutions=archive.sample_elites(4)["solution"], batch_size=4, seed=123), then Scheduler(archive, emitters). Since the archive is already populated, ask samples elites and adds independent Gaussian noise with sigma 0.1; initial_solutions is only used when empty. GaussianEmitter inherits the base no-op tell. The loop updates first when scheduled, then ask → reshape → evaluate_lsi → update best → tell → log each iteration. 200 iterations, batch 4; stated runtime ~45 min GPU, saved run 14:21. Logged QD score is normalized, unlike the story log. best is updated but never printed, so max objective is not found. [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

- Final image archive heatmap uses equal aspect, color limits 0–100, axes Diversity Metric 1 and Diversity Metric 2. The notebook then samples a 4×4 collage across measure-space regions, generating black tiles for empty regions. Its text describes smaller objects upper left and larger bottom right; dark sky at top and dusty sky at bottom. These are post-hoc visible trends, not supervised axis labels. [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

### Inferences
- Suggested walkthrough order: solution → objective/measures → archive cell competition → emitter proposal → scheduler cycle → logged progress → archive examples. This is an editorial ordering derived from the notebook sections, not a paper claim. [Stories](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb), [images](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb).

### Gaps
- Exact per-iteration maximum objective: not found in either saved log. Numerical image-coordinate bounds: not printed. Hardware for the saved image runtime: not found.

## Logged evidence and exact notebook cells

### Takeaway
The following tables retain every saved metric line; the complete text/code transcript preserves exact prompts and examples. Each cell is linked to its release-tag notebook.

### Cited Findings

#### qdaif metric history

| Iteration | Coverage % | QD score | Max objective | Source |
|---:|---:|---:|---|---|
| 5 | 1.750 | 51.800 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb) |
| 10 | 2.750 | 85.800 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb) |
| 15 | 4.000 | 125.800 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb) |
| 20 | 4.750 | 148.200 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb) |
| 25 | 5.250 | 165.600 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb) |
| 30 | 6.000 | 190.000 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb) |
| 35 | 7.250 | 232.000 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb) |
| 40 | 8.250 | 267.600 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb) |
| 45 | 8.750 | 285.600 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb) |
| 50 | 9.750 | 318.200 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb) |
| 55 | 11.000 | 361.800 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb) |
| 60 | 11.750 | 388.000 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb) |
| 65 | 12.500 | 414.600 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb) |
| 70 | 13.500 | 449.800 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb) |
| 75 | 14.000 | 467.200 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb) |
| 80 | 14.750 | 491.200 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb) |
| 85 | 15.750 | 527.000 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb) |
| 90 | 16.750 | 562.400 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb) |
| 95 | 17.500 | 588.400 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb) |
| 100 | 18.000 | 606.400 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb) |
| 105 | 18.750 | 628.800 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb) |
| 110 | 19.250 | 646.000 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb) |
| 115 | 19.750 | 659.400 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb) |
| 120 | 19.750 | 659.800 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb) |
| 125 | 20.500 | 686.600 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb) |
| 130 | 21.250 | 704.800 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb) |
| 135 | 22.250 | 739.400 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb) |
| 140 | 23.000 | 765.600 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb) |
| 145 | 24.000 | 795.600 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb) |
| 150 | 24.500 | 816.000 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb) |
| 155 | 24.500 | 816.000 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb) |
| 160 | 25.000 | 833.000 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb) |
| 165 | 25.250 | 842.000 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb) |
| 170 | 25.750 | 860.200 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb) |
| 175 | 26.250 | 877.600 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb) |
| 180 | 26.750 | 896.000 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb) |
| 185 | 27.250 | 913.400 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb) |
| 190 | 27.500 | 921.400 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb) |
| 195 | 28.000 | 938.600 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb) |
| 200 | 28.500 | 955.800 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb) |

#### qdaif full text and code transcript

##### Cell 0 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
# Orchestrating LLMs to Write Diverse Stories with Quality Diversity through AI Feedback

by [Bryon Tjanaka](https://btjanaka.net/); edited and reviewed by [Andrew Dai](https://daia99.github.io/), [Saeed Hedayatian](https://conflictednerd.github.io/), and [Sid Srikanth](https://sidsrikanth.com/)

_This tutorial is part of the series of pyribs tutorials! See [here](https://docs.pyribs.org/en/stable/tutorials.html) for the list of all tutorials and the order in which they should be read._

Given a creative writing task like "write a story about a spy and a politician," there are many possible interpretations. For instance, we could write a story that ends with the spy getting away with classified information. Or one where the spy and politician set aside their differences and team up to overthrow a government. Or even one where the spy and politician fall in love and leave their lives behind. In short, a wide range of stories exist, each with their own unique plot and details.

<figure style="width: 50%; margin-left: auto; margin-right: auto;">

![](_static/spy-and-politician.png)

<figcaption style="text-align: center; font-style: italic">"An image of a suspicious spy and a rich politician." Generated with ChatGPT.</figcaption>
</figure>

To explore the range of such possibilities, [Quality Diversity through AI Feedback (QDAIF; Bradley 2024)](https://qdaif.github.io/) proposes to orchestrate LLMs in two ways. First, QDAIF uses LLMs to _generate_ new stories. Given a story, QDAIF prompts the LLM to mutate the story into a new one. Second, and equally as important, QDAIF builds on the insight that _LLMs can operate in highly subjective search spaces like creative writing and code generation_. Namely, QDAIF leverages LLMs to _evaluate_ each story, by determining whether the story is high-quality and whether it is diverse compared to other stories. In other words, the LLMs compute the _objective_ and _measure_ functions of a quality diversity problem. Thus, QDAIF can repeatedly generate and evaluate stories, eventually producing an archive of diverse stories.

In this tutorial, we will demonstrate how to implement a variation of QDAIF in pyribs on the task of writing a story about a spy and a politician. We will show how to evaluate each story, set up the QD algorithm components, run the algorithm, and visualize the results. We hope that this tutorial will help connect folks in the QD and LLM communities by making it easier to implement algorithms that combine these two methods.

_Since this tutorial involves running LLMs, we recommend running it on a machine with a GPU, either on Colab or on a local workstation. It is possible to run on CPU on a standard laptop, although it will be much slower. Alternatively, if you would like to use an API such as OpenAI or Google Gemini, it should be possible to set that up (more details below)._
```

##### Cell 1 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
## Setup

First, let's set up the prerequisites for this tutorial.
```

##### Cell 2 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
### Python Dependencies

In addition to pyribs, this tutorial depends on [LangChain](https://python.langchain.com/docs/introduction/), a framework for developing LLM applications, and `langchain-ollama`, the package that integrates LangChain with Ollama.
```

##### Cell 3 (code)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```python
from __future__ import annotations

%pip install ribs[visualize] langchain langchain-ollama tqdm itables "moviepy>=2.0.0"
```

##### Cell 4 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
### Instantiating an LLM with LangChain and Ollama

To make this tutorial flexible to the choice of LLM, we use [LangChain](https://python.langchain.com/docs/introduction/). Among other things, LangChain provides a common interface for operating with LLMs from providers like OpenAI and Google. In this tutorial, we will use LangChain's integration with [Ollama](https://ollama.com). Ollama is a framework that enables efficiently running LLMs on local machines. In other words, _we will use LangChain to call an LLM hosted locally by Ollama_.

If you are running this tutorial on your own machine, please follow the [installation instructions](https://ollama.com/download) for Ollama and skip the cell below (but execute the ones after it). If you are running on Google Colab, install Ollama using the cell below, which was adapted from this [notebook](https://colab.research.google.com/github/5aharsh/collama/blob/main/Ollama_Setup.ipynb) by Saharsh Anand.

**Note:** If you would like to use an LLM from an API like OpenAI or Google Gemini, LangChain also provides integrations for many APIs; more details (such as how to use `init_chat_model`) are available [here](https://python.langchain.com/docs/tutorials/llm_chain/). In that case, feel free to skip this section and instantiate a `model` variable on your own. Note that we assume the model is a _chat model_, i.e., an instance of [BaseChatModel](https://python.langchain.com/api_reference/core/language_models/langchain_core.language_models.chat_models.BaseChatModel.html).
```

##### Cell 5 (code)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```python
!sudo apt update
!sudo apt install -y pciutils
!curl -fsSL https://ollama.com/install.sh | sh
```

##### Cell 6 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
After installing Ollama, we start the Ollama server in the background. If Ollama is already running, this cell will output an error that the address is already in use. _This is perfectly fine; simply proceed to the next steps._
```

##### Cell 7 (code)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```python
import subprocess
import threading
import time


def run_ollama_serve() -> None:
    subprocess.Popen(["ollama", "serve"])


thread = threading.Thread(target=run_ollama_serve)
thread.start()
time.sleep(5)  # Wait for the server to start.
```

Saved output: [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
Error: listen tcp 127.0.0.1:11434: bind: address already in use

```

##### Cell 8 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
We can now pull the LLM from Ollama's library and instantiate it in LangChain. We have chosen [Llama 3.1](https://ollama.com/library/llama3.1:8b-instruct-q4_K_M), specifically the 8B parameter version finetuned for instruction following. We choose the `q4_K_M` [quantization](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md) as it is a recommended size that balances between speed/memory usage and accuracy. For alternative models, visit the Ollama library [here](https://ollama.com/library). One example alternative is `llama3.1:70b-instruct-q4_K_M` (70B version of Llama 3.1).
```

##### Cell 9 (code)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```python
from langchain_ollama import ChatOllama

model_name = "llama3.1:8b-instruct-q4_K_M"  # @param ["llama3.1:8b-instruct-q4_K_M","llama3.1:70b-instruct-q4_K_M"] {"allow-input":true}

# Pull the model from the Ollama library.
!ollama pull {model_name}
print("---")

# Instantiate the model in LangChain.
model = ChatOllama(model=model_name)
print("Model:", model)
```

##### Cell 10 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
## Problem Statement

In this tutorial, we frame the problem of writing stories as a _quality diversity (QD) optimization_ problem. QD seeks to find a diverse set of high-performing solutions/stories (we use "solutions" and "stories" interchangeably in this tutorial; "solutions" is the more general term). Performance is defined in terms of an _objective_, and diversity is defined with respect to a vector-valued _measure_ function. The solutions in the set should individually maximize the objective while collectively attaining every possible output of the measure function.

We define the following objective and measures. These definitions are adapted from the Stories domain in the original QDAIF paper (Bradley 2023), where the measures are slightly different.

- **Objective:** Is the story a high-quality short story about a suspicious spy and a rich politician?
- **Measure 0:** Is the story a romance story?
- **Measure 1:** Does the story have a happy ending?

To illustrate, our final set of solutions may contain stories like:

- A story about a spy and a policitican that is a _romance_ story with a _happy_ ending.
- A story about a spy and a policitican that is _not a romance_ story with an _unhappy_ ending.
- A story about a spy and a policitican that is _not a romance_ story with an ending that is _neither happy nor sad_.
```

##### Cell 11 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
## AI Feedback with an Evaluator

The first ingredient for QDAIF is an evaluator that calls the LLM to evaluate the objective and measures defined in the problem statement above. For each objective and measure, we write a prompt (which we will show later) and pass it to the LLM. From there, we can extract the LLM's score for each prompt in one of many ways. For example, the original QDAIF paper predominantly extracts scores by asking the LLM to output an answer like yes or no and analyzing the logits of the tokens associated with that answer. Notably, this approach requires access to the logits output by the LLM, and logits are not always available. Thus, in this tutorial, we instead ask the LLM to output a rating on a scale of 1 to 10 when evaluating the objective and each measure. This approach is general in that it works with any LLM, but a key drawback is that the rating is stochastic and requires multiple evaluations.

The `Evaluator` class below implements this approach. Note the `n_evals` parameter that determines how many times to evaluate the objective and each measure. The final score for the objective and each measure is averaged over the ratings from the `n_evals` evaluations.
```

##### Cell 12 (code)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```python
from collections.abc import Collection

import numpy as np
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.base import Runnable
from pydantic import BaseModel, Field


class Evaluator:
    """Manages an LLM to compute the objective and measures.

    Args:
        model: Chat model for computing evaluations.
        objective_prompt: Prompt for the objective.
        measure_0_prompt: Prompt for the first measure (measure 0).
        measure_1_prompt: Prompt for the second measure (measure 1).
        n_evals: Number of times to evaluate the objective and each measure.
    """

    def __init__(
        self,
        *,
        model: BaseChatModel,
        objective_prompt: str,
        measure_0_prompt: str,
        measure_1_prompt: str,
        n_evals: int,
    ) -> None:
        self.model = model
        self.n_evals = n_evals
        self.min_score = 1
        self.max_score = 10

        # To receive the output from the LLM in a consistent format, we use structured
        # output (https://python.langchain.com/docs/how_to/structured_output/). This
        # Pydantic model defines the schema for receiving ratings from the LLM. Note
        # that the text in the schema class (including class name, field name, field
        # description, docstrings) all have some influence on the LLM output.
        class Rating(BaseModel):
            rating: int = Field(description="The rating on a scale of 1 to 10.")

        # Objective. We first define a chat template, where the `objective_prompt`
        # passed in is the system prompt, and the `text` of the story is the user's
        # message. Then, we form a chain that connects this template to the model. We
        # do the same for measure 0 and measure 1 below. For more background on
        # LangChain, refer to the documentation, such as:
        # - https://python.langchain.com/docs/tutorials/llm_chain/
        # - https://python.langchain.com/docs/concepts/lcel/
        self.objective_template = ChatPromptTemplate(
            [("system", objective_prompt), ("user", "{text}")]
        )
        self.objective_chain = (
            self.objective_template | self.model.with_structured_output(Rating)
        )

        # Measure 0.
        self.measure_0_template = ChatPromptTemplate(
            [("system", measure_0_prompt), ("user", "{text}")]
        )
        self.measure_0_chain = (
            self.measure_0_template | self.model.with_structured_output(Rating)
        )

        # Measure 1.
        self.measure_1_template = ChatPromptTemplate(
            [("system", measure_1_prompt), ("user", "{text}")]
        )
        self.measure_1_chain = (
            self.measure_1_template | self.model.with_structured_output(Rating)
        )

    def compute_score(
        self, chain: Runnable, texts: Collection[str]
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Uses the given chain to compute scores for the given batch of input texts.

        Each text input is evaluated `n_evals` times.

        Two values are returned:
        - The first value is `all_scores`, which is a list where each entry contains the
          `n_evals` scores for each text.
        - The second is `mean_scores`, which is the mean score for each piece of text.
        """
        inputs = [{"text": text} for text in texts for _ in range(self.n_evals)]
        outputs = chain.batch(inputs)

        all_scores = []
        mean_scores = []

        for i in range(0, len(outputs), self.n_evals):
            results = outputs[i : i + self.n_evals]
            scores = []
            for r in results:
                # Note: this assumes the schema for each result has a `rating` field,
                # which may not be the case if you modify the schema above.
                score = np.clip(r.rating, self.min_score, self.max_score)
                scores.append(score)

            scores = np.asarray(scores)
            all_scores.append(scores)
            mean_scores.append(scores.mean())

        return all_scores, np.asarray(mean_scores)

    def evaluate(self, texts: Collection[str]) -> tuple[np.ndarray, np.ndarray]:
        objectives = self.compute_score(self.objective_chain, texts)[1]
        measure_0 = self.compute_score(self.measure_0_chain, texts)[1]
        measure_1 = self.compute_score(self.measure_1_chain, texts)[1]
        measures = np.stack((measure_0, measure_1), axis=1)
        return objectives, measures
```

##### Cell 13 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
Having defined the `Evaluator`, we now instantiate it with the `model` (defined above in Setup), the prompts for the objective and measures, and the number of evaluations `n_evals`. Each prompt starts with defining a characteristic of the story and ends with asking the LLM to output a rating from 1 to 10. Feel free to customize the prompts to specify different objectives and measures!
```

##### Cell 14 (code)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```python
evaluator = Evaluator(
    model=model,
    objective_prompt="Determine if the input text contains a high-quality short story containing two characters, a suspicious spy, and a rich politician. For example, a high-quality short story would have good flow, interesting plot, and not repeat similar sentences or undesired items such as titles and URLs. Output your rating on a scale of 1 to 10. Output only a single integer.",
    measure_0_prompt="Determine if the input text is a romance story. For example, a romance story talks about two characters who fall in love with each other. Output your rating on a scale of 1 to 10. Output only a single integer.",
    measure_1_prompt="Determine if the input text is a story with a happy ending. For example, a story where the two characters make peace with each other has a happy ending. Output your rating on a scale of 1 to 10. Output only a single integer.",
    n_evals=5,
)
```

##### Cell 15 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
As an example, let's use the `Evaluator` to evaluate two stories.
```

##### Cell 16 (code)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```python
objectives, measures = evaluator.evaluate(
    [
        "The rich politician, Tom's life took a turn for the worst - he feared all of his close aides all of a sudden after sensing danger in his clique. There was a civil war going on, and he feared for his life. One day, one of his security guards, turned secret agent, decided to sneak into the classified files room, and spied on Johnny, who was in the room. He wanted to find Johnny's weakness, and strike at the right time.",
        "Jack was a politician in the city when one day he met Sarah. Sarah had been working for the government as a secret spy. Jack decided he really liked Sarah, and they fell in love. They both quite their jobs and decided to live in the countryside together.",
    ]
)

for i, (obj, meas) in enumerate(zip(objectives, measures, strict=True)):
    print(f"Story {i} | Objective: {obj}, Measure 0: {meas[0]}, Measure 1: {meas[1]}")
```

Saved output: [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
Story 0 | Objective: 6.6, Measure 0: 1.8, Measure 1: 2.4
Story 1 | Objective: 7.4, Measure 0: 8.0, Measure 1: 9.2

```

##### Cell 17 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
## QDAIF Components in pyribs

QDAIF builds on the MAP-Elites framework ([Mouret 2015](https://arxiv.org/abs/1504.04909)). Broadly speaking, MAP-Elites maintains an archive for storing the solutions to a QD problem. During execution, it selects solutions from this archive, mutates them, and inserts them back into the archive. In pyribs, this workflow is implemented with three components: an archive, emitters, and a scheduler. Below we describe each component and its functionality.
```

##### Cell 18 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
### GridArchive for Storing Stories

The archive for QDAIF is a [`GridArchive`](https://docs.pyribs.org/en/stable/api/ribs.archives.GridArchive.html), which divides the measure space into a grid and stores a solution in each grid cell. When a solution is added to this archive, it is assigned to a grid cell based on its measure values. If there is already a solution in that cell, the archive only retains the solution that has the higher objective value.

Below, we specify a $20 \times 20$ grid (`dims`) where each dimension of the grid ranges from 1 to 10 (`ranges`). For those familiar with `GridArchive` from previous tutorials, several other settings are needed since this `GridArchive` stores text-based solutions, whereas previous tutorials involved continuous vectors. First, we set `solution_dim=()`, indicating a scalar solution. Second, we set `solution_dtype=object` so that the archive can store pieces of text, which are single objects of type `str` (i.e., "scalar" objects).
```

##### Cell 19 (code)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```python
from ribs.archives import GridArchive

archive = GridArchive(
    solution_dim=(),
    dims=[20, 20],
    ranges=[(1, 10), (1, 10)],
    solution_dtype=object,
    objective_dtype=np.float32,
    measures_dtype=np.float32,
)
```

##### Cell 20 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
### Custom Emitter for Generating New Stories

In pyribs, emitters generate new candidate solutions for evaluation. In the case of QDAIF, the emitter generates new stories by randomly selecting existing stories in the archive and mutating them. The original QDAIF performed such mutations with Language Model Crossover from [Meyerson 2023](https://arxiv.org/abs/2302.12170). In this tutorial, we differ from the original QDAIF by implementing "directional" variations inspired by PLAN-QD in [Srikanth 2025](https://arxiv.org/abs/2504.03991).

Specifically, the `LLMDirectionalEmitter` below does the following when its `ask()` method is called. First, it selects a random story from the archive. Next, for each measure, it selects a random direction, either up or down. For instance, for measure 0 in this tutorial, the story can be made to sound more like a romance story or less like a romance story, and for measure 1, the ending of the story can be made more happy or less happy. Having selected a story and these two random directions (one for each measure), the emitter calls the LLM to mutate the story in the given directions, e.g., "given this story, make the story sound more like a romance story, and make the ending more happy." The result is a new story that is then sent for evaluation.

Finally, note that the `LLMDirectionalEmitter` also takes in an `initial_solutions` parameter. These define the initial texts that will be output by the emitter on the first iteration.
```

##### Cell 21 (code)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```python
from ribs.archives import ArchiveBase
from ribs.emitters import EmitterBase


class LLMDirectionalEmitter(EmitterBase):
    """Uses LLMs to modify pieces of text in random archive directions.

    Args:
        archive: Archive of solutions, e.g., :class:`ribs.archives.GridArchive`. The
            archive must contain solutions of type :class:`str`.
        model: LLM for mutating pieces of text.
        mutation_prompt: Prompt for mutating the text. This prompt must contain fields
            for ``measure_0_direction`` and ``measure_1_direction``.
        measure_*_{down,up}: Prompts indicating directions for mutating the story. Note
            that these directions are inserted into the ``mutation_prompt``.
        batch_size: Number of solutions to return in :meth:`ask`. Note that on the first
            iteration, ``initial_solutions`` will be returned, which may not be of
            length ``batch_size``.
        initial_solutions: Initial pieces of text for the LLM.
        seed: Value to seed the random number generator. Set to None to avoid a fixed
            seed.
    """

    def __init__(
        self,
        archive: ArchiveBase,
        *,
        model: BaseChatModel,
        mutation_prompt: str,
        measure_0_down: str,
        measure_0_up: str,
        measure_1_down: str,
        measure_1_up: str,
        batch_size: int,
        initial_solutions: Collection[str],
        seed: int | None = None,
    ) -> None:
        EmitterBase.__init__(
            self,
            archive,
            solution_dim=archive.solution_dim,
            bounds=None,
            lower_bounds=None,
            upper_bounds=None,
        )

        self.model = model
        self.batch_size = batch_size
        self.initial_solutions = initial_solutions
        self.rng = np.random.default_rng(seed)

        # This template provides the mutation prompt as the LLM's system prompt, and
        # then provides the story as the `text` field.
        self.mutation_template = ChatPromptTemplate(
            [("system", mutation_prompt), ("user", "{text}")]
        )

        # Each measure has two possible directions: up or down.
        self.mutation_dirs = {
            "measure_0": [measure_0_down, measure_0_up],
            "measure_1": [measure_1_down, measure_1_up],
        }

        # Schema for retrieving stories from the model.
        class Story(BaseModel):
            story: str = Field(description="The modified story.")

        self.mutation_chain = (
            self.mutation_template | self.model.with_structured_output(Story)
        )

    def ask(self) -> list[str]:
        """The ask method returns a list of new solutions/stories for evaluation."""
        # When no solutions have been created yet, just return the initial seed texts.
        if self.archive.empty:
            return self.initial_solutions

        # During normal operation, sample texts and directions to pass to the LLM.
        prompts = []
        for _ in range(self.batch_size):
            # For both measure_0 and measure_1 (hence size=2), choose between the two
            # possible directions.
            dirs = self.rng.choice(2, size=2)

            prompts.append(
                {
                    "text": self._archive.sample_elites(1)["solution"][0],
                    "measure_0_direction": self.mutation_dirs["measure_0"][dirs[0]],
                    "measure_1_direction": self.mutation_dirs["measure_1"][dirs[1]],
                }
            )

        stories = self.mutation_chain.batch(prompts)
        return [s.story for s in stories]
```

##### Cell 22 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
Below, we instantiate the emitter with prompts to enable making the story more/less romantic and the ending more/less happy. We also provide three initial stories (`initial_solutions`) drawn from the QDAIF paper. Feel free to modify any of the prompts or the initial stories.

To save on computation, we have only defined one emitter with a batch size of 1, so we only generate 1 solution per iteration. We also use the same `model` that is used by the `Evaluator`, so that we do not have to load another model. It is certainly possible to increase the batch size of each emitter, create multiple emitters that each perform mutations with different prompts, or use different models than that used in the `Evaluator`. If you have the time, we encourage trying out these ideas!
```

##### Cell 23 (code)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```python
emitters = [
    LLMDirectionalEmitter(
        archive,
        model=model,
        mutation_prompt="The following is a story about two characters, a suspicious spy, and a rich politician. Modify the story in the following ways: {measure_0_direction}, and {measure_1_direction}. Output only the new story.",
        measure_0_down="make the story sound less like a romance story",
        measure_0_up="make the story sound more like a romance story",
        measure_1_down="make the ending of the story less happy",
        measure_1_up="make the ending of the story more happy",
        batch_size=1,
        initial_solutions=[
            # From QDAIF paper (Appendix A.21).
            "A spy named Joanne wants to infiltrate the premises of Karl Johnson, a highly-influential figure in the city. Karl was a wealthy mayor, and would do anything in his power to suppress any opposing voices. Joanne wanted to figure out what Karl was hiding, but she took a turn for the worse, as she was highly suspicious in her presence outside his home.",
            "The wealthy entrepreneur and member of parliament, Susan, hosted a party at her mansion. She invited all of the residents, as well as an unusual looking man. The man, Dave, was wearing a tacky shirt, and star-shaped glasses, and was actually a spy. He made the whole room laugh with his jokes, and had a secret agenda - to find what Susan does in her private fun room!",
            "The rich politician, Tom's life took a turn for the worst - he feared all of his close aides all of a sudden after sensing danger in his clique. There was a civil war going on, and he feared for his life. One day, one of his security guards, turned secret agent, decided to sneak into the classified files room, and spied on Johnny, who was in the room. He wanted to find Johnny's weakness, and strike at the right time.",
        ],
    ),
    # Additional emitters would go here (`emitters` is a list of emitters).
]
```

##### Cell 24 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
### Scheduler for Composing QDAIF

To compose the archive and emitters for QDAIF together, we instantiate a [`Scheduler`](https://docs.pyribs.org/en/stable/api/ribs.schedulers.Scheduler.html). The scheduler handles the QD algorithm logic like calling the archive and emitters.
```

##### Cell 25 (code)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```python
from ribs.schedulers import Scheduler

scheduler = Scheduler(archive, emitters)
```

##### Cell 26 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
## Visualizing the QDAIF Archive

Before we run QDAIF with the scheduler, we must introduce one last ingredient that will help understand the outputs of the algorithm. Using [`grid_archive_heatmap`](https://docs.pyribs.org/en/stable/api/ribs.visualize.grid_archive_heatmap.html) from the [`ribs.visualize`](https://docs.pyribs.org/en/stable/api/ribs.visualize.html) module, we can plot a heatmap of the archive that represents all the stories generated. The axes correspond to the two measures, while the color of each cell shows the objective. For now, the heatmap is empty, but it will fill up as QDAIF progresses.
```

##### Cell 27 (code)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```python
from pathlib import Path

import matplotlib.pyplot as plt

from ribs.visualize import grid_archive_heatmap


def plot_heatmap(
    archive: GridArchive, show: bool = False, file: str | Path | None = None
) -> None:
    """Plots the archive.

    Pass `show` to display the heatmap with plt.show().
    Pass `file` to save to the given file.
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    grid_archive_heatmap(archive, ax=ax, vmin=1, vmax=10)
    ax.set_title("A story about a suspicious spy and a rich politician.")
    ax.set_xlabel("Romance Story")
    ax.set_ylabel("Happy Ending")
    ax.set_aspect("equal")
    fig.tight_layout()
    if file is not None:
        fig.savefig(file)
    if show:
        plt.show()
    plt.close(fig)
```

##### Cell 28 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
Below, we also set up a logging directory in `qdaif_output/`. **Note that this cell will also remove any existing `qdaif_output/` directory!**
```

##### Cell 29 (code)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```python
import shutil

output_dir = Path("qdaif_output/")
shutil.rmtree(output_dir, ignore_errors=True)
output_dir.mkdir()
```

##### Cell 30 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
And now, we plot the initial heatmap:
```

##### Cell 31 (code)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```python
plot_heatmap(archive, show=True, file=output_dir / f"heatmap_{0:06d}.png")
```

Saved output: [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
<Figure size 800x600 with 2 Axes>
```

##### Cell 32 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
## Running QDAIF

With the components created, we can execute QDAIF in the loop below. During this loop, we first call `ask` on the `scheduler`, which generates new stories (`solutions`) by calling the `ask` method defined on `LLMDirectionalEmitter` above. After evaluating the `objectives` and `measures` of the stories, we pass them to the `scheduler` with `tell`, which inserts the stories into the `archive`. Along the way, we regularly log metrics and save the archive data as a CSV.

On a T4 GPU on Colab, this loop should take **~1 hour** to run. It was also tested locally on an RTX A6000 GPU, where it took around half an hour to run.
```

##### Cell 33 (code)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```python
import sys

from tqdm import tqdm, trange

total_itrs = 200

for itr in trange(1, total_itrs + 1, file=sys.stdout, desc="Iterations"):
    solutions = scheduler.ask()
    objectives, measures = evaluator.evaluate(solutions)
    scheduler.tell(objectives, measures)

    if itr % 5 == 0 or itr == total_itrs:
        tqdm.write(
            f"Iteration {itr:5d} | "
            f"Archive Coverage: {archive.stats.coverage:6.3%}  "
            f"QD Score: {archive.stats.qd_score:6.3f}"
        )

        # Save the archive's data as a CSV, which makes it easy to read the stories.
        archive.data(return_type="pandas").to_csv(output_dir / "archive.csv")

        # Plot the heatmap in the output directory defined earlier. To avoid crowding
        # the logs, we show the heatmap much less frequently than we plot it.
        plot_heatmap(
            archive,
            show=itr % 50 == 0 or itr == total_itrs,
            file=output_dir / f"heatmap_{itr:06d}.png",
        )
```

Saved output: [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
Iteration     5 | Archive Coverage:  1.750%  QD Score: 51.800                                 
Iteration    10 | Archive Coverage:  2.750%  QD Score: 85.800                                 
Iteration    15 | Archive Coverage:  4.000%  QD Score: 125.800                                
Iteration    20 | Archive Coverage:  4.750%  QD Score: 148.200                                
Iteration    25 | Archive Coverage:  5.250%  QD Score: 165.600                                
Iteration    30 | Archive Coverage:  6.000%  QD Score: 190.000                                
Iteration    35 | Archive Coverage:  7.250%  QD Score: 232.000                                
Iteration    40 | Archive Coverage:  8.250%  QD Score: 267.600                                
Iteration    45 | Archive Coverage:  8.750%  QD Score: 285.600                                
Iteration    50 | Archive Coverage:  9.750%  QD Score: 318.200                                
Iterations:  24%|██████████▊                                 | 49/200 [07:35<20:08,  8.00s/it]
```

Saved output: [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
<Figure size 800x600 with 2 Axes>
```

Saved output: [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
Iteration    55 | Archive Coverage: 11.000%  QD Score: 361.800                                
Iteration    60 | Archive Coverage: 11.750%  QD Score: 388.000                                
Iteration    65 | Archive Coverage: 12.500%  QD Score: 414.600                                
Iteration    70 | Archive Coverage: 13.500%  QD Score: 449.800                                
Iteration    75 | Archive Coverage: 14.000%  QD Score: 467.200                                
Iteration    80 | Archive Coverage: 14.750%  QD Score: 491.200                                
Iteration    85 | Archive Coverage: 15.750%  QD Score: 527.000                                
Iteration    90 | Archive Coverage: 16.750%  QD Score: 562.400                                
Iteration    95 | Archive Coverage: 17.500%  QD Score: 588.400                                
Iteration   100 | Archive Coverage: 18.000%  QD Score: 606.400                                
Iterations:  50%|█████████████████████▊                      | 99/200 [17:21<20:05, 11.93s/it]
```

Saved output: [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
<Figure size 800x600 with 2 Axes>
```

Saved output: [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
Iteration   105 | Archive Coverage: 18.750%  QD Score: 628.800                                
Iteration   110 | Archive Coverage: 19.250%  QD Score: 646.000                                
Iteration   115 | Archive Coverage: 19.750%  QD Score: 659.400                                
Iteration   120 | Archive Coverage: 19.750%  QD Score: 659.800                                
Iteration   125 | Archive Coverage: 20.500%  QD Score: 686.600                                
Iteration   130 | Archive Coverage: 21.250%  QD Score: 704.800                                
Iteration   135 | Archive Coverage: 22.250%  QD Score: 739.400                                
Iteration   140 | Archive Coverage: 23.000%  QD Score: 765.600                                
Iteration   145 | Archive Coverage: 24.000%  QD Score: 795.600                                
Iteration   150 | Archive Coverage: 24.500%  QD Score: 816.000                                
Iterations:  74%|████████████████████████████████           | 149/200 [24:00<07:16,  8.56s/it]
```

Saved output: [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
<Figure size 800x600 with 2 Axes>
```

Saved output: [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
Iteration   155 | Archive Coverage: 24.500%  QD Score: 816.000                                
Iteration   160 | Archive Coverage: 25.000%  QD Score: 833.000                                
Iteration   165 | Archive Coverage: 25.250%  QD Score: 842.000                                
Iteration   170 | Archive Coverage: 25.750%  QD Score: 860.200                                
Iteration   175 | Archive Coverage: 26.250%  QD Score: 877.600                                
Iteration   180 | Archive Coverage: 26.750%  QD Score: 896.000                                
Iteration   185 | Archive Coverage: 27.250%  QD Score: 913.400                                
Iteration   190 | Archive Coverage: 27.500%  QD Score: 921.400                                
Iteration   195 | Archive Coverage: 28.000%  QD Score: 938.600                                
Iteration   200 | Archive Coverage: 28.500%  QD Score: 955.800                                
Iterations: 100%|██████████████████████████████████████████▊| 199/200 [35:20<00:18, 18.22s/it]
```

Saved output: [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
<Figure size 800x600 with 2 Axes>
```

Saved output: [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
Iterations: 100%|███████████████████████████████████████████| 200/200 [35:21<00:00, 10.61s/it]

```

##### Cell 34 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
## Final Archive Heatmap and Video

Below we show the final heatmap of the archive.
```

##### Cell 35 (code)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```python
plot_heatmap(archive, show=True, file=None)
```

Saved output: [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
<Figure size 800x600 with 2 Axes>
```

##### Cell 36 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
During the run, we also plotted heatmaps of the archive and saved them in the `output_dir`.
```

##### Cell 37 (code)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```python
!ls {output_dir}
```

Saved output: [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
archive.csv	    heatmap_000050.png	heatmap_000105.png  heatmap_000160.png
heatmap_000000.png  heatmap_000055.png	heatmap_000110.png  heatmap_000165.png
heatmap_000005.png  heatmap_000060.png	heatmap_000115.png  heatmap_000170.png
heatmap_000010.png  heatmap_000065.png	heatmap_000120.png  heatmap_000175.png
heatmap_000015.png  heatmap_000070.png	heatmap_000125.png  heatmap_000180.png
heatmap_000020.png  heatmap_000075.png	heatmap_000130.png  heatmap_000185.png
heatmap_000025.png  heatmap_000080.png	heatmap_000135.png  heatmap_000190.png
heatmap_000030.png  heatmap_000085.png	heatmap_000140.png  heatmap_000195.png
heatmap_000035.png  heatmap_000090.png	heatmap_000145.png  heatmap_000200.png
heatmap_000040.png  heatmap_000095.png	heatmap_000150.png  heatmap_video.mp4
heatmap_000045.png  heatmap_000100.png	heatmap_000155.png

```

##### Cell 38 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
Now, we can connect them all into a single video with `moviepy`.
```

##### Cell 39 (code)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```python
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

# We must use `str` because moviepy expects the image paths to be str.
clip = ImageSequenceClip(sorted(map(str, output_dir.glob("heatmap_*.png"))), fps=6)

# Save video in output_dir and display in this notebook.
clip.write_videofile(str(output_dir / "heatmap_video.mp4"))
clip.display_in_notebook()
```

Saved output: [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
MoviePy - Building video qdaif_output/heatmap_video.mp4.
MoviePy - Writing video qdaif_output/heatmap_video.mp4


```

Saved output: [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
                                                                                              
```

Saved output: [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
MoviePy - Done !
MoviePy - video ready qdaif_output/heatmap_video.mp4
MoviePy - Building video __temp__.mp4.
MoviePy - Writing video __temp__.mp4


```

Saved output: [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
                                                                                              
```

Saved output: [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
MoviePy - Done !
MoviePy - video ready __temp__.mp4

```

Saved output: [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
<moviepy.video.io.display_in_notebook.HTML2 object>
```

##### Cell 40 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
## Reading Generated Stories

Having visualized the archive, we can take a look at some of the individual stories. The function below retrieves a story from the archive with the given measures and prints it out. Specifically, the [`retrieve_single`](https://docs.pyribs.org/en/stable/api/ribs.archives.GridArchive.html#ribs.archives.GridArchive.retrieve_single) method locates the cell with the given measures and returns the elite in that cell if it exists.
```

##### Cell 41 (code)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```python
import textwrap

from numpy.typing import ArrayLike


def retrieve_and_print(archive: ArchiveBase, measures: ArrayLike) -> None:
    occupied, elite = archive.retrieve_single(measures)

    if occupied:
        print(f"Objective: {elite['objective']:.3f}")
        print(f"Measure 0 (Romance Story): {elite['measures'][0]:.3f}")
        print(f"Measure 1 (Happy Ending): {elite['measures'][1]:.3f}")
        print("-" * 20)
        print(textwrap.fill(elite["solution"], 80))
    else:
        print("No solution available with the given measures.")
```

##### Cell 42 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
First, we can look at some stories that had high ratings for being both a romance story and having a happy ending.
```

##### Cell 43 (code)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```python
retrieve_and_print(archive, [9, 10])
```

Saved output: [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
Objective: 8.800
Measure 0 (Romance Story): 8.800
Measure 1 (Happy Ending): 9.600
--------------------
With every step he took through the grand halls of Green Industries, Karl
Johnson couldn't help but feel a flutter in his chest as he caught glimpses of
Harrison Green. The CEO's chiseled features and piercing gaze seemed to capture
him in their depths, leaving Karl feeling both intrigued and wary. Hired by
Harrison himself to investigate allegations of financial impropriety within the
company, Karl had been warned about the dangers of getting too close to the
billionaire. But as he delved deeper into the labyrinthine world of corporate
finance, he found himself increasingly drawn to the enigmatic CEO. With each
passing day, he became more aware of Harrison's calculating gaze that followed
him, and yet, he couldn't help but feel a spark whenever their eyes met or their
hands touched in passing. As they worked together to unravel the tangled threads
of corruption within the company, Karl found himself torn between his duty as a
spy and his growing attraction to Harrison. He knew he should be cautious, that
Harrison was off-limits, but he couldn't deny the sense of belonging he felt
whenever they stood close. One night, as they stood on the rooftop overlooking
the city skyline, Harrison turned to him with a hint of vulnerability in his
voice and confessed that he had been searching for someone like Karl – someone
who saw beyond the façade and understood the complexity beneath. Karl's heart
skipped a beat as Harrison took a step closer, their faces inches apart. The air
was charged with tension, but instead of suspicion or mistrust, Karl felt a
sense of comfort wash over him. He knew in that moment that he had found
something far more valuable than just answers – he had found a connection, a
bond that went beyond mere duty. And as Harrison's lips brushed against his own,
the world around them melted away, leaving only the thrumming beat of their
hearts and the promise of a love yet to be written. As they pulled back, Karl
smiled up at him, feeling a sense of peace settle over him. Harrison's eyes
crinkled at the corners as he smiled back, his face alight with warmth. In that
moment, Karl knew that he had found his home – not just in Green Industries, but
in Harrison's heart.

```

##### Cell 44 (code)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```python
retrieve_and_print(archive, [8, 8.5])
```

Saved output: [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
Objective: 9.000
Measure 0 (Romance Story): 8.000
Measure 1 (Happy Ending): 8.600
--------------------
The civil war raging outside seemed a distant hum compared to the tension
building within Senator Tom's inner circle. He couldn't shake off the feeling
that one of them was hiding something from him, waiting for the perfect moment
to strike. That's when he met Emily, his new security guard, with eyes as sharp
and piercing as the stars on a clear night. Her gaze scanned Tom's inner circle
with an intensity that sent shivers down his spine. As Tom entrusted her with
the highest level of clearance, Emily began to unravel the threads of suspicion
surrounding Johnny, one of Tom's most trusted aides. But it was more than just
loyalty or ambition driving Johnny's actions - Emily saw a deep-seated pain
lurking beneath his surface, a pain that echoed the turmoil she'd seen in
countless others during her years as a spy. She knew better than to trust
anyone, but something about Johnny's vulnerability pricked at her instincts. As
she delved deeper into the world of Tom's inner circle, Emily found herself
questioning everything - including her own motivations for being there. Was it
truly just to protect Tom from harm, or was there more at play? And what did she
feel when she thought of him? A spark, like a flicker of electricity in the
dark, flared within her. As she gazed into his eyes, Emily felt the air thick
with tension, and for a moment, they simply stared at each other, the world
around them melting away. But Emily knew that in her line of work, sparks often
led to flames that consumed everything in their path. She needed to keep her
wits about her, not get caught up in emotions. As she continued to dig deeper,
Emily's findings took a disastrous turn. Johnny was indeed hiding something -
and it wasn't just his loyalty to Tom that was in question. It was his
allegiance to the enemy. The revelation sent shockwaves through the inner
circle, leaving Tom reeling and Emily questioning her own judgment. Just as all
hope seemed lost, Emily's quick thinking foiled an attempted assassination on
Tom's life. In the aftermath of the failed attempt, Tom realized that his
suspicions about one of his aides were not unfounded and that he had made a
mistake in trusting them. He also discovered that Johnny was not working alone
but was part of a larger scheme to take down his family's reputation. Tom was
grateful for Emily's bravery and quick thinking. In recognition of her efforts,
Tom decided to offer Emily a full-time position on his security team and even
allowed her to lead the investigation into Johnny's betrayal. With Johnny behind
bars and the threat to Tom's life foiled, Emily finally felt like she'd found a
new purpose. She knew that in order to protect Tom from harm, she needed to put
aside her distrust of people and learn to trust those around her. As they stood
together, facing the challenges ahead, Emily felt a sense of belonging for the
first time in years. The tension between them was still palpable, but this time
it wasn't just about attraction - it was about camaraderie and shared purpose.

```

##### Cell 45 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
Next, we can look at stories that were rated low on both measures.
```

##### Cell 46 (code)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```python
retrieve_and_print(archive, [1.8, 2.8])
```

Saved output: [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
Objective: 8.800
Measure 0 (Romance Story): 1.600
Measure 1 (Happy Ending): 3.000
--------------------
Karl Johnson's mansion loomed before Joanne, its grand entrance a testament to
his wealth and influence. A skilled operative, she had been tasked with
infiltrating Karl's inner circle to uncover the corruption he perpetuated. As
she stood outside, her eyes scanned the perimeter for any signs of security
breaches or potential escape routes. Her mission was clear: extract evidence,
expose Karl's schemes, and bring him to justice. With a practiced air, Joanne
made her way up the steps, her hand resting on the door handle as if preparing
to leave at a moment's notice. The evening sun cast long shadows across the
lawn, but she didn't let it faze her. Her focus remained laser-sharp, attuned to
every detail. Inside, Karl emerged from his study, a man of power and politics.
Their eyes met briefly, and for an instant, they exchanged a cold, calculated
stare. No spark of attraction here; only two minds locked in a battle of wits.
Joanne's training kicked in as she assessed the situation, her mind racing with
strategies and contingency plans. She knew Karl was cunning and had a reputation
for ruthlessness. But so did she. The air around them vibrated with tension, but
they both held back, waiting to see who would make the first move. As the sun
dipped below the horizon, Joanne stepped forward, her eyes never leaving Karl's
face. Her hand closed into a fist as she mentally prepared for the long,
grueling night ahead. Karl raised an eyebrow, his expression unreadable, but
Joanne detected a hint of curiosity beneath the surface. She knew that in this
game of cat and mouse, only one could emerge victorious. With her wits and
training guiding her, Joanne took another step forward, ready to face whatever
lay ahead. As their eyes locked once more, a silent understanding passed between
them: the battle for dominance had begun.

```

##### Cell 47 (code)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```python
retrieve_and_print(archive, [3.0, 2.0])
```

Saved output: [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
Objective: 8.600
Measure 0 (Romance Story): 3.000
Measure 1 (Happy Ending): 2.200
--------------------
It was supposed to be just another charity gala at Susan's sprawling mansion.
But for Alex, a seasoned spy with a talent for sniffing out secrets, this event
had all the makings of a high-stakes mission. And he wasn't alone in his
suspicions - Jack, a wealthy businessman and close friend of Susan's, seemed
equally on edge as they mingled with the other guests.

```

##### Cell 48 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
And finally, one that was rated as a romance story but not with a happy ending.
```

##### Cell 49 (code)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```python
retrieve_and_print(archive, [8.5, 2.5])
```

Saved output: [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
Objective: 8.800
Measure 0 (Romance Story): 8.400
Measure 1 (Happy Ending): 2.400
--------------------
Lena's eyes locked onto Harrison Grant across the room, his chiseled features
and piercing gaze impossible to ignore even from afar. She felt a flutter in her
chest as she recalled their previous encounters, their charged chemistry
undeniable even beneath the polished veneer of their social interactions. But
tonight was about more than just pleasure - it was about business, and Lena's
latest mission had brought her to this lavish gala with a single goal: to
uncover the secrets Harrison kept hidden behind his gilded facade.

```

##### Cell 50 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
## Listing All Stories in a Table

Using the [`data`](https://docs.pyribs.org/en/stable/api/ribs.archives.GridArchive.html#ribs.archives.GridArchive.data) method, we can retrieve a dataframe with all the data in the archive. Then, we can display this table with [itables](https://mwouts.github.io/itables/).
```

##### Cell 51 (code)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```python
import itables

itables.init_notebook_mode(all_interactive=False)

df = archive.data(return_type="pandas")

# See here for more info on `classes`: https://datatables.net/manual/styling/classes
# The `columnDefs` sets the `solution` column (i.e., the target is 0, which is the
# first column) to have left-aligned text ("dt-body-left") -- see Cell classes here:
# https://datatables.net/manual/styling/classes#Cell-classes and this issue here:
# https://github.com/mwouts/itables/issues/22
itables.show(
    df,
    classes="display",
    columnDefs=[{"className": "dt-body-left", "targets": 0}],
)
```

Saved output: [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
<IPython.core.display.HTML object>
```

Saved output: [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
<IPython.core.display.HTML object>
```

Saved output: [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
<IPython.core.display.HTML object>
```

##### Cell 52 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
## Conclusion

There has been a growing number of works at the intersection of QD and LLMs, with QDAIF being one of the first to leverage this synergy. We hope this tutorial will enhance the accessibility of such methods! We envision making it easy for LLM folks to use QD, and for QD folks to use LLMs.

This tutorial showed just one application of such methods, for generating stories about "a suspicious spy and a rich politician." However, there are many further applications, such as for [red-teaming LLMs](https://sites.google.com/view/rainbow-teaming). In this tutorial itself, there is also ample room for extension. For example, you may have noticed that some stories have issues with coherence, where they jump around or change certain details. Furthermore, while most stories have nice prose, many are incomplete, in that they lack a coherent beginning, middle, and end. Finally, the ratings associated with the stories are occasionally inaccurate, e.g., some are rated as having a happy ending when the ending sounds quite sad. We hypothesize that many of these issues stem from the inherent quirks of working with LLMs, and perhaps tuning the various prompts, increasing the number of evaluations (`n_evals`), or even just running for longer (200 iterations is very few) may help. We encourage you to try things out for yourself and see what works!
```

##### Cell 53 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb)

```text
## Citation

If you find this tutorial useful, please cite it as:

```
@article{pyribs_qdaif,
  title   = {Orchestrating LLMs to Write Diverse Stories with Quality Diversity through AI Feedback},
  author  = {Bryon Tjanaka},
  journal = {pyribs.org},
  year    = {2025},
  url     = {https://docs.pyribs.org/en/stable/tutorials/qdaif.html}
}
```

QDAIF can also be cited as:

```
@inproceedings{
  bradley2024qualitydiversity,
  title = {Quality-Diversity through {AI} Feedback},
  author = {Herbie Bradley and Andrew Dai and Hannah Benita Teufel and Jenny Zhang and Koen Oostermeijer and Marco Bellagente and Jeff Clune and Kenneth Stanley and Gregory Schott and Joel Lehman},
  booktitle = {The Twelfth International Conference on Learning Representations},
  year = {2024},
  url = {https://openreview.net/forum?id=owokKCrGYr}
}
```
```


#### qdhf metric history

| Iteration | Coverage % | QD score | Max objective | Source |
|---:|---:|---:|---|---|
| 1 | 34.5 | 24.97 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 2 | 35.25 | 25.52 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 3 | 36.0 | 26.06 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 4 | 36.25 | 26.24 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 5 | 36.75 | 26.6 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 6 | 37.25 | 26.97 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 7 | 37.75 | 27.33 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 8 | 38.5 | 27.86 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 9 | 38.75 | 28.03 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 10 | 39.0 | 28.2 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 11 | 39.5 | 28.57 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 12 | 39.5 | 28.58 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 13 | 40.0 | 28.94 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 14 | 40.25 | 29.12 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 15 | 41.0 | 29.67 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 16 | 41.25 | 29.85 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 17 | 41.5 | 30.04 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 18 | 41.75 | 30.21 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 19 | 42.0 | 30.39 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 20 | 42.0 | 30.39 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 21 | 29.5 | 21.38 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 22 | 29.75 | 21.57 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 23 | 30.5 | 22.11 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 24 | 31.0 | 22.49 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 25 | 31.25 | 22.67 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 26 | 32.0 | 23.23 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 27 | 32.75 | 23.76 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 28 | 33.25 | 24.13 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 29 | 33.5 | 24.3 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 30 | 34.5 | 25.03 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 31 | 34.5 | 25.03 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 32 | 35.5 | 25.75 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 33 | 35.75 | 25.92 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 34 | 36.25 | 26.29 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 35 | 36.75 | 26.66 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 36 | 37.75 | 27.39 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 37 | 38.0 | 27.57 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 38 | 38.5 | 27.94 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 39 | 39.25 | 28.48 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 40 | 39.5 | 28.65 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 41 | 40.0 | 29.02 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 42 | 40.75 | 29.57 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 43 | 41.25 | 29.92 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 44 | 41.5 | 30.09 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 45 | 42.0 | 30.44 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 46 | 42.25 | 30.63 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 47 | 42.5 | 30.81 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 48 | 42.5 | 30.81 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 49 | 42.75 | 31.01 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 50 | 42.75 | 31.01 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 51 | 30.0 | 21.81 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 52 | 30.0 | 21.81 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 53 | 30.25 | 22.0 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 54 | 30.75 | 22.37 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 55 | 31.25 | 22.72 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 56 | 32.0 | 23.25 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 57 | 32.25 | 23.43 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 58 | 32.75 | 23.8 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 59 | 33.0 | 23.98 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 60 | 33.0 | 23.98 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 61 | 33.5 | 24.34 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 62 | 33.75 | 24.51 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 63 | 34.0 | 24.69 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 64 | 34.25 | 24.87 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 65 | 35.0 | 25.41 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 66 | 36.0 | 26.13 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 67 | 36.25 | 26.31 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 68 | 37.0 | 26.86 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 69 | 37.25 | 27.05 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 70 | 37.5 | 27.22 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 71 | 37.75 | 27.4 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 72 | 38.0 | 27.58 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 73 | 38.5 | 27.94 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 74 | 38.5 | 27.94 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 75 | 38.75 | 28.13 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 76 | 39.0 | 28.31 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 77 | 39.25 | 28.5 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 78 | 39.75 | 28.86 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 79 | 40.5 | 29.42 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 80 | 40.75 | 29.6 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 81 | 41.0 | 29.78 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 82 | 41.75 | 30.33 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 83 | 41.75 | 30.33 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 84 | 42.0 | 30.51 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 85 | 42.25 | 30.69 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 86 | 42.5 | 30.87 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 87 | 42.5 | 30.87 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 88 | 42.5 | 30.87 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 89 | 43.0 | 31.25 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 90 | 43.25 | 31.43 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 91 | 43.5 | 31.62 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 92 | 43.5 | 31.62 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 93 | 44.0 | 31.98 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 94 | 44.5 | 32.35 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 95 | 44.5 | 32.35 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 96 | 44.5 | 32.35 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 97 | 44.5 | 32.35 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 98 | 45.0 | 32.71 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 99 | 45.25 | 32.88 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 100 | 45.5 | 33.07 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 101 | 32.0 | 23.27 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 102 | 32.25 | 23.44 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 103 | 32.5 | 23.63 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 104 | 32.75 | 23.81 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 105 | 33.25 | 24.16 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 106 | 33.5 | 24.34 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 107 | 33.75 | 24.53 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 108 | 34.0 | 24.71 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 109 | 34.25 | 24.9 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 110 | 34.5 | 25.09 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 111 | 35.0 | 25.43 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 112 | 35.5 | 25.79 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 113 | 36.0 | 26.16 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 114 | 36.5 | 26.53 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 115 | 37.0 | 26.92 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 116 | 37.5 | 27.27 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 117 | 38.25 | 27.83 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 118 | 38.5 | 28.01 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 119 | 39.0 | 28.38 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 120 | 39.25 | 28.57 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 121 | 39.75 | 28.93 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 122 | 40.25 | 29.3 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 123 | 40.75 | 29.66 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 124 | 41.25 | 30.03 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 125 | 41.75 | 30.39 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 126 | 41.75 | 30.4 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 127 | 41.75 | 30.4 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 128 | 42.0 | 30.57 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 129 | 42.5 | 30.92 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 130 | 43.0 | 31.28 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 131 | 43.5 | 31.65 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 132 | 44.25 | 32.19 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 133 | 44.5 | 32.37 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 134 | 44.75 | 32.56 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 135 | 45.0 | 32.74 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 136 | 45.25 | 32.92 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 137 | 45.5 | 33.1 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 138 | 45.5 | 33.11 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 139 | 45.75 | 33.29 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 140 | 46.25 | 33.64 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 141 | 46.5 | 33.82 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 142 | 46.5 | 33.83 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 143 | 46.5 | 33.83 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 144 | 46.5 | 33.83 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 145 | 46.75 | 34.01 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 146 | 47.0 | 34.2 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 147 | 47.5 | 34.56 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 148 | 47.5 | 34.56 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 149 | 47.75 | 34.75 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 150 | 47.75 | 34.75 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 151 | 48.25 | 35.11 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 152 | 48.5 | 35.28 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 153 | 48.75 | 35.46 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 154 | 49.0 | 35.65 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 155 | 49.25 | 35.84 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 156 | 49.5 | 36.02 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 157 | 49.75 | 36.22 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 158 | 50.0 | 36.4 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 159 | 50.0 | 36.42 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 160 | 50.5 | 36.79 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 161 | 50.74999999999999 | 36.97 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 162 | 50.74999999999999 | 36.97 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 163 | 51.0 | 37.15 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 164 | 51.0 | 37.15 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 165 | 51.24999999999999 | 37.33 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 166 | 51.74999999999999 | 37.69 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 167 | 51.74999999999999 | 37.69 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 168 | 51.74999999999999 | 37.69 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 169 | 52.0 | 37.88 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 170 | 52.0 | 37.89 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 171 | 52.25 | 38.07 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 172 | 52.25 | 38.07 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 173 | 52.75 | 38.44 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 174 | 52.75 | 38.45 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 175 | 53.0 | 38.63 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 176 | 53.0 | 38.65 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 177 | 53.0 | 38.67 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 178 | 53.25 | 38.85 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 179 | 53.5 | 39.04 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 180 | 53.5 | 39.04 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 181 | 53.75 | 39.21 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 182 | 54.0 | 39.4 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 183 | 54.0 | 39.4 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 184 | 54.50000000000001 | 39.76 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 185 | 55.00000000000001 | 40.11 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 186 | 55.00000000000001 | 40.11 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 187 | 55.00000000000001 | 40.11 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 188 | 55.25 | 40.3 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 189 | 55.25 | 40.3 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 190 | 55.25 | 40.3 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 191 | 55.50000000000001 | 40.48 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 192 | 56.00000000000001 | 40.82 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 193 | 56.25 | 41.0 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 194 | 56.49999999999999 | 41.17 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 195 | 56.75 | 41.35 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 196 | 56.75 | 41.36 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 197 | 57.25 | 41.73 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 198 | 57.25 | 41.73 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 199 | 57.25 | 41.74 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |
| 200 | 57.49999999999999 | 41.93 | not found | [notebook](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb) |

Retraining logs, in schedule order: [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```text
Iterations:   0%|                               | 0/200 [00:00<?, ?it/s]0.7s (14 epochs) | DivProj (n=1000) fitted with val acc.: 0.74
Iterations:  10%|██▏                   | 20/200 [02:33<06:45,  2.25s/it]1.0s (15 epochs) | DivProj (n=2000) fitted with val acc.: 0.774
Iterations:  25%|█████▌                | 50/200 [05:13<04:47,  1.92s/it]2.2s (15 epochs) | DivProj (n=3000) fitted with val acc.: 0.8066666666666666
Iterations:  50%|██████████▌          | 100/200 [08:42<03:11,  1.92s/it]1.6s (12 epochs) | DivProj (n=4000) fitted with val acc.: 0.8
```

#### qdhf full text and code transcript

##### Cell 0 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```text
# Incorporating Human Feedback into Quality Diversity for Diversified Text-to-Image Generation

by [Li Ding](https://liding.info), author of [Quality Diversity through Human Feedback (QDHF)](https://liding.info/qdhf/); edited and reviewed by [Bryon Tjanaka](https://btjanaka.net/)

*This tutorial is part of the series of pyribs tutorials! See [here](https://docs.pyribs.org/en/stable/tutorials.html) for the list of all tutorials and the order in which they should be read.*
```

##### Cell 1 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```text
Foundation models such as large language models (LLMs) and text-to-image generation models ([Stable Diffusion](https://arxiv.org/abs/2112.10752), [DALL·E](https://openai.com/dall-e-3), etc.) in effect compress vast archives of human knowledge into powerful and flexible tools, serving as a foundation for down-stream applications. One mechanism to build upon such foundational knowledge is reinforcement learning from human feedback (RLHF), which can make models both easier to use (by aligning them to human instructions), and more competent (by improving their capabilities based on human preferences).

RLHF is a relatively new paradigm, and its deployments often follow the relatively narrow recipe of maximizing a learned reward model of averaged human preferences over model responses. However, there are drawbacks when it is used to optimize for average human preferences, especially in generative tasks that demand diverse and creative model responses. On the other hand, Quality Diversity (QD) algorithms excel at identifying diverse and high-quality solutions but often rely on manually-crafted diversity metrics, as shown in previous examples, such as [DQD](https://docs.pyribs.org/en/stable/tutorials/tom_cruise_dqd.html).

In this work, we introduce [Quality Diversity through Human Feedback (QDHF)](https://liding.info/qdhf/), which aims to broaden the RLHF recipe to include optimizing for interesting diversity among responses, which is of practical importance for many creative applications such as text-to-image generation. Such increased diversity can also improve optimization for complex and open-ended tasks (through improved exploration), personalization (serving individual rather than average human preference), and fairness (to offset algorithmic biases in gender, ethnicity, and more).

<br>

![Overview of QDHF.](_static/qdhf-method.png)

<br>

Our main idea is to derive distinct representations of what humans find interestingly different, and use such diversity representations to support optimization. An overview of our method is shown in the diagram above.

In this tutorial, we implement QDHF for the latent space illumination (LSI) experiment. LSI uses QD algorithms to search for a diverse collection of images in a GAN’s latent space, which has been introduced in the previous [DQD tutorial](https://docs.pyribs.org/en/stable/tutorials/tom_cruise_dqd.html). However, instead of generating images of a celebrity, this LSI pipeline uses [Stable Diffusion](https://arxiv.org/abs/2112.10752) to generate diverse images according to a given text prompt, like the one shown below (*prompt: an image of a bear in a national park*).

![QDHF example](_static/qdhf-example.jpg)

Another difference in this tutorial is that, QDHF uses human feedback to derive meaningful diversity metrics, not by manually specifying attributes such as age and hair length. We use a preference model, [DreamSim](https://dreamsim-nights.github.io/), which is trained on a dataset of human judgment of similarity between images, to source feedback in QDHF optimization. Specifically, we implement QDHF with Map-Elites, and run it with the Stable Diffusion+CLIP pipeline.

*This tutorial assumes that you are familiar with Latent Space Illumination (LSI). If you are not yet familiar with it, we recommend reviewing the [DQD tutorial](https://docs.pyribs.org/en/stable/tutorials/tom_cruise_dqd.html) which generates Tom Cruise Images or the [LSI MNIST tutorial](https://docs.pyribs.org/en/stable/tutorials/lsi_mnist.html) which generates diverse MNIST digits.*
```

##### Cell 2 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```text
## Setup

Since Stable Diffusion, CLIP, and DreamSim are fairly large models, you will need a GPU to run this tutorial. On Colab, activate the GPU by clicking "Runtime" in the toolbar at the top. Then, click "Change Runtime Type", and select "GPU".

Below, we check what GPU has been provided. The possible GPUs (at the time of writing) are as follows; factory reset the runtime if you do not have a desired GPU.

* V100 = Excellent (*Available only for Colab Pro users*)
* P100 = Very Good
* T4 = Good (*preferred*)
* K80 = Meh
* P4 = (*Not Recommended*)
```

##### Cell 3 (code)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```python
!nvidia-smi -L
```

Saved output: [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```text
GPU 0: NVIDIA RTX A6000 (UUID: GPU-76b34a40-ef0b-e82b-762b-652c7fb7e93c)

```

##### Cell 4 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```text
Now, we install various libraries, including pyribs, CLIP, Stable Diffusion, and DreamSim. This cell will take around 5-10 minutes to run since it involves downloading multiple large files.
```

##### Cell 5 (code)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```python
%pip install ribs[visualize] torch torchvision git+https://github.com/openai/CLIP tqdm dreamsim diffusers accelerate
```

##### Cell 6 (code)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```python
import sys
import time

import numpy as np
from tqdm import tqdm, trange
```

##### Cell 7 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```text
Let's first set up CUDA for PyTorch. We use float16 to save GPU memories for large models.
```

##### Cell 8 (code)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```python
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
print("Torch device:", DEVICE)

# Use float16 for GPU, float32 for CPU.
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
print("Torch dtype:", TORCH_DTYPE)
```

Saved output: [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```text
Torch device: cuda
Torch dtype: torch.float16

```

##### Cell 9 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```text
## Preliminaries: CLIP, Stable Diffusion, and DreamSim

Below we set up CLIP, Stable Diffusion, and DreamSim. The code for these models is adopted from the [DQD tutorial](https://docs.pyribs.org/en/stable/tutorials/tom_cruise_dqd.html), the [DreamSim demo](https://colab.research.google.com/drive/1taEOMzFE9g81D9AwH27Uhy2U82tQGAVI?usp=sharing), and the [Hugging Face Stable Diffusion example](https://huggingface.co/stabilityai/stable-diffusion-2-1). Each cell may take around 3 minutes to run since it involves downloading multiple large files.

*Since this section is focused on the LSI pipeline rather than on pyribs, feel free to skim past and come back to it later.*
```

##### Cell 10 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```text
### CLIP for Connecting Text and Images
```

##### Cell 11 (code)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```python
import clip

# This tutorial uses ViT-B/32, you may use other checkpoints depending on your resources and need.
CLIP_MODEL, CLIP_PREPROCESS = clip.load("ViT-B/32", device=DEVICE)
CLIP_MODEL.eval()
for p in CLIP_MODEL.parameters():
    p.requires_grad_(False)


def compute_clip_scores(imgs, text, return_clip_features=False):
    """Computes CLIP scores for a batch of images and a given text prompt."""
    img_tensor = torch.stack([CLIP_PREPROCESS(img) for img in imgs]).to(DEVICE)
    tokenized_text = clip.tokenize([text]).to(DEVICE)
    img_logits, _text_logits = CLIP_MODEL(img_tensor, tokenized_text)
    img_logits = img_logits.detach().cpu().numpy().astype(np.float32)[:, 0]
    img_logits = 1 / img_logits * 100
    # Remap the objective from minimizing [0, 10] to maximizing [0, 100]
    img_logits = (10.0 - img_logits) * 10.0

    if return_clip_features:
        clip_features = CLIP_MODEL.encode_image(img_tensor).to(TORCH_DTYPE)
        return img_logits, clip_features
    else:
        return img_logits
```

##### Cell 12 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```text
### DreamSim for Simulating Human Feedback
```

##### Cell 13 (code)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```python
from dreamsim import dreamsim

DREAMSIM_MODEL, DREAMSIM_PREPROCESS = dreamsim(
    pretrained=True, dreamsim_type="open_clip_vitb32", device=DEVICE
)
```

Saved output: [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```text
/home/tjanaka/pyribs/env/lib/python3.12/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
  from .autonotebook import tqdm as notebook_tqdm

```

Saved output: [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```text
Downloading checkpoint

```

Saved output: [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```text
100%|████████████████████████████████| 313M/313M [00:04<00:00, 65.9MB/s]

```

Saved output: [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```text
Unzipping...

```

Saved output: [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```text
/home/tjanaka/pyribs/env/lib/python3.12/site-packages/peft/tuners/tuners_utils.py:196: UserWarning: Already found a `peft_config` attribute in the model. This will lead to having multiple adapters in the model. Make sure to know what you are doing!
  warnings.warn(

```

##### Cell 14 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```text
### Stable Diffusion for Generating Images
By default, we use the [`miniSD-diffusers` checkpoint](https://huggingface.co/lambdalabs/miniSD-diffusers), which generates images at 256x256 for faster inference and less GPU memory usage. Please switch to the `stable-diffusion-2-1-base` checkpoint to generate images with higher resolution and quality.
```

##### Cell 15 (code)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```python
from diffusers import StableDiffusionPipeline

IMG_WIDTH = 256
IMG_HEIGHT = 256
SD_IN_HEIGHT = 32
SD_IN_WIDTH = 32
SD_CHECKPOINT = "lambdalabs/miniSD-diffusers"

# IMG_WIDTH = 512
# IMG_HEIGHT = 512
# SD_IN_HEIGHT = 64
# SD_IN_WIDTH = 64
# SD_CHECKPOINT = "stabilityai/stable-diffusion-2-1-base"

BATCH_SIZE = 4
SD_IN_CHANNELS = 4
SD_IN_SHAPE = (
    BATCH_SIZE,
    SD_IN_CHANNELS,
    SD_IN_HEIGHT,
    SD_IN_WIDTH,
)

SDPIPE = StableDiffusionPipeline.from_pretrained(
    SD_CHECKPOINT,
    torch_dtype=TORCH_DTYPE,
    safety_checker=None,  # For faster inference.
    requires_safety_checker=False,
)

SDPIPE.set_progress_bar_config(disable=True)
SDPIPE = SDPIPE.to(DEVICE)
```

Saved output: [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```text
Fetching 13 files: 100%|████████████████| 13/13 [00:36<00:00,  2.79s/it]
Loading pipeline components...:   0%|             | 0/6 [00:00<?, ?it/s]`torch_dtype` is deprecated! Use `dtype` instead!
Loading pipeline components...:  17%|▊    | 1/6 [00:00<00:01,  2.85it/s]An error occurred while trying to fetch /home/tjanaka/.cache/huggingface/hub/models--lambdalabs--miniSD-diffusers/snapshots/26ed8a9bfbf76f46a6cf60517dde321f900c44ce/unet: Error no file named diffusion_pytorch_model.safetensors found in directory /home/tjanaka/.cache/huggingface/hub/models--lambdalabs--miniSD-diffusers/snapshots/26ed8a9bfbf76f46a6cf60517dde321f900c44ce/unet.
Defaulting to unsafe serialization. Pass `allow_pickle=False` to raise an error instead.
Loading pipeline components...:  33%|█▋   | 2/6 [00:00<00:01,  2.00it/s]An error occurred while trying to fetch /home/tjanaka/.cache/huggingface/hub/models--lambdalabs--miniSD-diffusers/snapshots/26ed8a9bfbf76f46a6cf60517dde321f900c44ce/vae: Error no file named diffusion_pytorch_model.safetensors found in directory /home/tjanaka/.cache/huggingface/hub/models--lambdalabs--miniSD-diffusers/snapshots/26ed8a9bfbf76f46a6cf60517dde321f900c44ce/vae.
Defaulting to unsafe serialization. Pass `allow_pickle=False` to raise an error instead.
Loading pipeline components...: 100%|█████| 6/6 [00:01<00:00,  5.44it/s]

```

##### Cell 16 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```text
## Aligning Diversity Metrics with Human Preferences

Next, we define a latent projection model to characterize diversity from image features through constrastive learning from human feedback.
The `DivProj` model projects the image features (e.g., CLIP features with 512-d) onto a low-dimensional latent space (e.g., 2-d), where each dimension represents a numeric diversity metric.
An intuitive example is that the model may learn "object size" as a metric, where low values mean the object is small and high values mean it is large.
```

##### Cell 17 (code)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```python
from torch import nn


class DivProj(nn.Module):
    def __init__(self, input_dim, latent_dim=2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=latent_dim),
        )

    def forward(self, x):
        """Get diversity representations."""
        x = self.proj(x)
        return x

    def calc_dis(self, x1, x2):
        """Calculate diversity distance as (squared) L2 distance."""
        x1 = self.forward(x1)
        x2 = self.forward(x2)
        return torch.sum(torch.square(x1 - x2), -1)

    def triplet_delta_dis(self, ref, x1, x2):
        """Calculate delta distance comparing x1 and x2 to ref."""
        x1 = self.forward(x1)
        x2 = self.forward(x2)
        ref = self.forward(ref)
        return torch.sum(torch.square(ref - x1), -1) - torch.sum(
            torch.square(ref - x2), -1
        )
```

##### Cell 18 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```text
### Constrastive Learning for Diversity Projection

For contrastive learning, we use a triplet loss to train the diversity projection model. This loss encourages the model to project similar images close to each other in the latent space, and dissimilar images far from each other. The preference labels are given by the DreamSim model, which is pre-trained on the human feedback on image similarity, so we will use it as the source of feedback here.

To avoid redundancy in computation, we first extract DreamSim features from the images, and then calculate the cosine similarity of DreamSim features. This is equivalent to using the DreamSim model directly to calculate the similarity.

The whole `fit_div_proj` process works as follows: The input data is with shape of `(n, 3, image_size, image_size)`, where n is the number of triplets. The second dimension of 3 represents the reference, positive, and negative images. We split the data into traning and validation sets, and then train the diversity projection model with the triplet loss. The output is a diversity projection model that projects image features onto a low-dimensional latent space, where each dimension represents a numeric diversity metric.
```

##### Cell 19 (code)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```python
# Triplet loss with margin 0.05.
# The binary preference labels are scaled to y = 1 or -1 for the loss, where y = 1 means x2 is more similar to ref than x1.
loss_fn = lambda y, delta_dis: torch.max(  # noqa: E731
    torch.tensor([0.0]).to(DEVICE), 0.05 - (y * 2 - 1) * delta_dis
).mean()


def fit_div_proj(inputs, dreamsim_features, latent_dim, batch_size=32):
    """Trains the DivProj model on ground-truth labels."""
    t = time.time()
    model = DivProj(input_dim=inputs.shape[-1], latent_dim=latent_dim)
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_pref_data = inputs.shape[0]
    ref = inputs[:, 0]
    x1 = inputs[:, 1]
    x2 = inputs[:, 2]

    n_train = int(n_pref_data * 0.75)
    n_val = n_pref_data - n_train

    # Split data into train and val.
    ref_train = ref[:n_train]
    x1_train = x1[:n_train]
    x2_train = x2[:n_train]
    ref_val = ref[n_train:]
    x1_val = x1[n_train:]
    x2_val = x2[n_train:]

    # Split DreamSim features into train and val.
    ref_dreamsim_features = dreamsim_features[:, 0]
    x1_dreamsim_features = dreamsim_features[:, 1]
    x2_dreamsim_features = dreamsim_features[:, 2]
    ref_gt_train = ref_dreamsim_features[:n_train]
    x1_gt_train = x1_dreamsim_features[:n_train]
    x2_gt_train = x2_dreamsim_features[:n_train]
    ref_gt_val = ref_dreamsim_features[n_train:]
    x1_gt_val = x1_dreamsim_features[n_train:]
    x2_gt_val = x2_dreamsim_features[n_train:]

    val_acc = []
    n_iters_per_epoch = max((n_train) // batch_size, 1)
    for epoch in range(200):
        for _ in range(n_iters_per_epoch):
            optimizer.zero_grad()

            idx = np.random.choice(n_train, batch_size)
            batch_ref = ref_train[idx].float()
            batch1 = x1_train[idx].float()
            batch2 = x2_train[idx].float()

            # Get delta distance from model.
            delta_dis = model.triplet_delta_dis(batch_ref, batch1, batch2)

            # Get preference labels from DreamSim features.
            gt_dis = torch.nn.functional.cosine_similarity(
                ref_gt_train[idx], x2_gt_train[idx], dim=-1
            ) - torch.nn.functional.cosine_similarity(
                ref_gt_train[idx], x1_gt_train[idx], dim=-1
            )
            gt = (gt_dis > 0).to(TORCH_DTYPE)

            loss = loss_fn(gt, delta_dis)
            loss.backward()
            optimizer.step()

        # Validate.
        n_correct = 0
        n_total = 0
        with torch.no_grad():
            idx = np.arange(n_val)
            batch_ref = ref_val[idx].float()
            batch1 = x1_val[idx].float()
            batch2 = x2_val[idx].float()
            delta_dis = model.triplet_delta_dis(batch_ref, batch1, batch2)
            pred = delta_dis > 0
            gt_dis = torch.nn.functional.cosine_similarity(
                ref_gt_val[idx], x2_gt_val[idx], dim=-1
            ) - torch.nn.functional.cosine_similarity(
                ref_gt_val[idx], x1_gt_val[idx], dim=-1
            )
            gt = gt_dis > 0
            n_correct += (pred == gt).sum().item()
            n_total += len(idx)

        acc = n_correct / n_total
        val_acc.append(acc)

        # Early stopping if val_acc does not improve for 10 epochs.
        if epoch > 10 and np.mean(val_acc[-10:]) < np.mean(val_acc[-11:-1]):
            break

    print(
        f"{np.round(time.time() - t, 1)}s ({epoch + 1} epochs) | DivProj (n={n_pref_data}) fitted with val acc.: {acc}"
    )

    return model.to(TORCH_DTYPE), acc
```

##### Cell 20 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```text
### Helper Functions to Compute Objectives and Measures
```

##### Cell 21 (code)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```python
def compute_diversity_measures(clip_features, diversity_model):
    with torch.no_grad():
        measures = diversity_model(clip_features).detach().cpu().numpy()
    return measures


def evaluate_lsi(
    latents,
    prompt,
    return_features=False,
    diversity_model=None,
):
    """Evaluates the objective of LSI for a batch of latents and a given text prompt."""
    images = SDPIPE(
        prompt,
        num_images_per_prompt=latents.shape[0],
        latents=latents,
        # num_inference_steps=1,  # For testing.
    ).images

    objs, clip_features = compute_clip_scores(
        images,
        prompt,
        return_clip_features=True,
    )

    images = torch.cat([DREAMSIM_PREPROCESS(img) for img in images]).to(DEVICE)
    dreamsim_features = DREAMSIM_MODEL.embed(images)

    if diversity_model is not None:
        measures = compute_diversity_measures(clip_features, diversity_model)
    else:
        measures = None

    if return_features:
        return objs, measures, clip_features, dreamsim_features
    else:
        return objs, measures
```

##### Cell 22 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```text
## QDHF with pyribs

To use QDHF to generate diverse collections of images, we will use the MAP-Elites algorithm, where the diversity metrics are progressively trained and fine-tuned during optimization.
```

##### Cell 23 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```text
### Algorithm Setup

To set up the MAP-Elites instance for QDHF, we will set up the following components. First, we create the [`GridArchive`](https://docs.pyribs.org/en/stable/api/ribs.archives.GridArchive.html). The archive is 20x20 and stores Stable Diffusion latent vectors. The dimensions of vectors are (BATCH, 4, H, W) where H=W=32 if generating 256x256 images with `miniSD`, or 64 for 512x512 images with `stable-diffusion-2-1-base`. Next, we create a [`GaussianEmitter`](https://docs.pyribs.org/en/stable/api/ribs.emitters.GaussianEmitter.html) that generates solutions by mutating existing archive solutions with Gaussian noise. Finally, we set up a [`Scheduler`](https://docs.pyribs.org/en/stable/api/ribs.schedulers.Scheduler.html) to combine everything together.
```

##### Cell 24 (code)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```python
# ruff: noqa: NPY002
from ribs.archives import GridArchive
from ribs.emitters import GaussianEmitter
from ribs.schedulers import Scheduler

GRID_SIZE = (20, 20)
SEED = 123
np.random.seed(SEED)
torch.manual_seed(SEED)


def tensor_to_list(tensor):
    sols = tensor.detach().cpu().numpy().astype(np.float32)
    return sols.reshape(sols.shape[0], -1)


def list_to_tensor(list_):
    sols = np.array(list_).reshape(
        len(list_), 4, SD_IN_HEIGHT, SD_IN_WIDTH
    )  # Hard-coded for now.
    return torch.tensor(sols, dtype=TORCH_DTYPE, device=DEVICE)


def create_scheduler(
    sols,
    objs,
    clip_features,
    diversity_model,
    seed=None,
):
    measures = compute_diversity_measures(clip_features, diversity_model)
    archive_bounds = np.array(
        [np.quantile(measures, 0.01, axis=0), np.quantile(measures, 0.99, axis=0)]
    ).T

    sols = tensor_to_list(sols)

    # Set up archive.
    archive = GridArchive(
        solution_dim=len(sols[0]), dims=GRID_SIZE, ranges=archive_bounds, seed=seed
    )

    # Add initial solutions to the archive.
    archive.add(sols, objs, measures)

    # Set up the GaussianEmitter.
    emitters = [
        GaussianEmitter(
            archive=archive,
            sigma=0.1,
            initial_solutions=archive.sample_elites(BATCH_SIZE)["solution"],
            batch_size=BATCH_SIZE,
            seed=seed,
        )
    ]

    # Return the archive and scheduler.
    return archive, Scheduler(archive, emitters)
```

##### Cell 25 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```text
## LSI with QDHF

Having set up the necessary components above, we now run QDHF to search for images of a given prompt. The default prompt is "an astronaut riding a horse on mars".

### Training

The execution loop is similar to the experiments in previous tutorials. The main difference is that when the iteration is in `update_schedule`, we will update the QD archive by fine-tuning the diversity metrics and adding current solutions to the archive again with regard to their updated diversity measures.

The total number of iterations is set to be 200, and it should take ~45min to run on a GPU.
```

##### Cell 26 (code)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```python
# Change here for whatever you would like to generate.
prompt = "an astronaut riding a horse on mars"

INIT_POP = 200  # Initial population.

TOTAL_ITRS = 200  # Total number of iterations.
update_schedule = [1, 21, 51, 101]  # Iterations on which to update the archive.
n_pref_data = 1000  # Number of preferences used in each update.

archive: GridArchive = None
best = 0.0
for itr in trange(1, TOTAL_ITRS + 1, file=sys.stdout, desc="Iterations"):
    itr_start = time.time()

    # Update archive and scheduler if needed.
    if itr in update_schedule:
        if archive is None:
            tqdm.write("Initializing archive and diversity projection.")

            all_sols = []
            all_clip_features = []
            all_dreamsim_features = []
            all_objs = []

            # Sample random solutions and get judgment on similarity.
            n_batches = INIT_POP // BATCH_SIZE
            for _ in range(n_batches):
                sols = torch.randn(SD_IN_SHAPE, device=DEVICE, dtype=TORCH_DTYPE)
                objs, _, clip_features, dreamsim_features = evaluate_lsi(
                    sols, prompt, return_features=True
                )
                all_sols.append(sols)
                all_clip_features.append(clip_features)
                all_dreamsim_features.append(dreamsim_features)
                all_objs.append(objs)
            all_sols = torch.concat(all_sols, dim=0)
            all_clip_features = torch.concat(all_clip_features, dim=0)
            all_dreamsim_features = torch.concat(all_dreamsim_features, dim=0)
            all_objs = np.concatenate(all_objs, axis=0)

            # Initialize the diversity projection model.
            div_proj_data = []
            div_proj_labels = []
            for _ in range(n_pref_data):
                idx = np.random.choice(all_sols.shape[0], 3)
                div_proj_data.append(all_clip_features[idx])
                div_proj_labels.append(all_dreamsim_features[idx])
            div_proj_data = torch.concat(div_proj_data, dim=0)
            div_proj_labels = torch.concat(div_proj_labels, dim=0)
            div_proj_data = div_proj_data.reshape(n_pref_data, 3, -1)
            div_proj_label = div_proj_labels.reshape(n_pref_data, 3, -1)
            diversity_model, div_proj_acc = fit_div_proj(
                div_proj_data,
                div_proj_label,
                latent_dim=2,
            )

        else:
            tqdm.write("Updating archive and diversity projection.")

            # Get all the current solutions and collect feedback.
            all_sols = list_to_tensor(archive.data("solution"))
            n_batches = np.ceil(len(all_sols) / BATCH_SIZE).astype(int)
            all_clip_features = []
            all_dreamsim_features = []
            all_objs = []
            for i in range(n_batches):
                sols = all_sols[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
                objs, _, clip_features, dreamsim_features = evaluate_lsi(
                    sols, prompt, return_features=True
                )
                all_clip_features.append(clip_features)
                all_dreamsim_features.append(dreamsim_features)
                all_objs.append(objs)
            all_clip_features = torch.concat(
                all_clip_features, dim=0
            )  # n_pref_data * 3, dim
            all_dreamsim_features = torch.concat(all_dreamsim_features, dim=0)
            all_objs = np.concatenate(all_objs, axis=0)

            # Update the diversity projection model.
            additional_features = []
            additional_labels = []
            for _ in range(n_pref_data):
                idx = np.random.choice(all_sols.shape[0], 3)
                additional_features.append(all_clip_features[idx])
                additional_labels.append(all_dreamsim_features[idx])
            additional_features = torch.concat(additional_features, dim=0)
            additional_labels = torch.concat(additional_labels, dim=0)
            additional_div_proj_data = additional_features.reshape(n_pref_data, 3, -1)
            additional_div_proj_label = additional_labels.reshape(n_pref_data, 3, -1)
            div_proj_data = torch.concat(
                (div_proj_data, additional_div_proj_data), dim=0
            )
            div_proj_label = torch.concat(
                (div_proj_label, additional_div_proj_label), dim=0
            )
            diversity_model, div_proj_acc = fit_div_proj(
                div_proj_data,
                div_proj_label,
                latent_dim=2,
            )

        archive, scheduler = create_scheduler(
            all_sols,
            all_objs,
            all_clip_features,
            diversity_model,
            seed=SEED,
        )

    # Primary QD loop.
    sols = scheduler.ask()
    sols = list_to_tensor(sols)
    objs, measures, clip_features, dreamsim_features = evaluate_lsi(
        sols, prompt, return_features=True, diversity_model=diversity_model
    )
    best = max(best, *objs)
    scheduler.tell(objs, measures)

    # This can be used as a flag to save on the final iteration, but note that
    # we do not save results in this tutorial.
    final_itr = itr == TOTAL_ITRS

    # Update the summary statistics for the archive.
    qd_score, coverage = archive.stats.norm_qd_score, archive.stats.coverage

    tqdm.write(f"QD score: {np.round(qd_score, 2)} Coverage: {coverage * 100}")
```

Saved output: [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```text
Initializing archive and diversity projection.                          
Iterations:   0%|                               | 0/200 [00:00<?, ?it/s]0.7s (14 epochs) | DivProj (n=1000) fitted with val acc.: 0.74
QD score: 24.97 Coverage: 34.5                                          
QD score: 25.52 Coverage: 35.25                                         
QD score: 26.06 Coverage: 36.0                                          
QD score: 26.24 Coverage: 36.25                                         
QD score: 26.6 Coverage: 36.75                                          
QD score: 26.97 Coverage: 37.25                                         
QD score: 27.33 Coverage: 37.75                                         
QD score: 27.86 Coverage: 38.5                                          
QD score: 28.03 Coverage: 38.75                                         
QD score: 28.2 Coverage: 39.0                                           
QD score: 28.57 Coverage: 39.5                                          
QD score: 28.58 Coverage: 39.5                                          
QD score: 28.94 Coverage: 40.0                                          
QD score: 29.12 Coverage: 40.25                                         
QD score: 29.67 Coverage: 41.0                                          
QD score: 29.85 Coverage: 41.25                                         
QD score: 30.04 Coverage: 41.5                                          
QD score: 30.21 Coverage: 41.75                                         
QD score: 30.39 Coverage: 42.0                                          
QD score: 30.39 Coverage: 42.0                                          
Updating archive and diversity projection.                              
Iterations:  10%|██▏                   | 20/200 [02:33<06:45,  2.25s/it]1.0s (15 epochs) | DivProj (n=2000) fitted with val acc.: 0.774
QD score: 21.38 Coverage: 29.5                                          
QD score: 21.57 Coverage: 29.75                                         
QD score: 22.11 Coverage: 30.5                                          
QD score: 22.49 Coverage: 31.0                                          
QD score: 22.67 Coverage: 31.25                                         
QD score: 23.23 Coverage: 32.0                                          
QD score: 23.76 Coverage: 32.75                                         
QD score: 24.13 Coverage: 33.25                                         
QD score: 24.3 Coverage: 33.5                                           
QD score: 25.03 Coverage: 34.5                                          
QD score: 25.03 Coverage: 34.5                                          
QD score: 25.75 Coverage: 35.5                                          
QD score: 25.92 Coverage: 35.75                                         
QD score: 26.29 Coverage: 36.25                                         
QD score: 26.66 Coverage: 36.75                                         
QD score: 27.39 Coverage: 37.75                                         
QD score: 27.57 Coverage: 38.0                                          
QD score: 27.94 Coverage: 38.5                                          
QD score: 28.48 Coverage: 39.25                                         
QD score: 28.65 Coverage: 39.5                                          
QD score: 29.02 Coverage: 40.0                                          
QD score: 29.57 Coverage: 40.75                                         
QD score: 29.92 Coverage: 41.25                                         
QD score: 30.09 Coverage: 41.5                                          
QD score: 30.44 Coverage: 42.0                                          
QD score: 30.63 Coverage: 42.25                                         
QD score: 30.81 Coverage: 42.5                                          
QD score: 30.81 Coverage: 42.5                                          
QD score: 31.01 Coverage: 42.75                                         
QD score: 31.01 Coverage: 42.75                                         
Updating archive and diversity projection.                              
Iterations:  25%|█████▌                | 50/200 [05:13<04:47,  1.92s/it]2.2s (15 epochs) | DivProj (n=3000) fitted with val acc.: 0.8066666666666666
QD score: 21.81 Coverage: 30.0                                          
QD score: 21.81 Coverage: 30.0                                          
QD score: 22.0 Coverage: 30.25                                          
QD score: 22.37 Coverage: 30.75                                         
QD score: 22.72 Coverage: 31.25                                         
QD score: 23.25 Coverage: 32.0                                          
QD score: 23.43 Coverage: 32.25                                         
QD score: 23.8 Coverage: 32.75                                          
QD score: 23.98 Coverage: 33.0                                          
QD score: 23.98 Coverage: 33.0                                          
QD score: 24.34 Coverage: 33.5                                          
QD score: 24.51 Coverage: 33.75                                         
QD score: 24.69 Coverage: 34.0                                          
QD score: 24.87 Coverage: 34.25                                         
QD score: 25.41 Coverage: 35.0                                          
QD score: 26.13 Coverage: 36.0                                          
QD score: 26.31 Coverage: 36.25                                         
QD score: 26.86 Coverage: 37.0                                          
QD score: 27.05 Coverage: 37.25                                         
QD score: 27.22 Coverage: 37.5                                          
QD score: 27.4 Coverage: 37.75                                          
QD score: 27.58 Coverage: 38.0                                          
QD score: 27.94 Coverage: 38.5                                          
QD score: 27.94 Coverage: 38.5                                          
QD score: 28.13 Coverage: 38.75                                         
QD score: 28.31 Coverage: 39.0                                          
QD score: 28.5 Coverage: 39.25                                          
QD score: 28.86 Coverage: 39.75                                         
QD score: 29.42 Coverage: 40.5                                          
QD score: 29.6 Coverage: 40.75                                          
QD score: 29.78 Coverage: 41.0                                          
QD score: 30.33 Coverage: 41.75                                         
QD score: 30.33 Coverage: 41.75                                         
QD score: 30.51 Coverage: 42.0                                          
QD score: 30.69 Coverage: 42.25                                         
QD score: 30.87 Coverage: 42.5                                          
QD score: 30.87 Coverage: 42.5                                          
QD score: 30.87 Coverage: 42.5                                          
QD score: 31.25 Coverage: 43.0                                          
QD score: 31.43 Coverage: 43.25                                         
QD score: 31.62 Coverage: 43.5                                          
QD score: 31.62 Coverage: 43.5                                          
QD score: 31.98 Coverage: 44.0                                          
QD score: 32.35 Coverage: 44.5                                          
QD score: 32.35 Coverage: 44.5                                          
QD score: 32.35 Coverage: 44.5                                          
QD score: 32.35 Coverage: 44.5                                          
QD score: 32.71 Coverage: 45.0                                          
QD score: 32.88 Coverage: 45.25                                         
QD score: 33.07 Coverage: 45.5                                          
Updating archive and diversity projection.                              
Iterations:  50%|██████████▌          | 100/200 [08:42<03:11,  1.92s/it]1.6s (12 epochs) | DivProj (n=4000) fitted with val acc.: 0.8
QD score: 23.27 Coverage: 32.0                                          
QD score: 23.44 Coverage: 32.25                                         
QD score: 23.63 Coverage: 32.5                                          
QD score: 23.81 Coverage: 32.75                                         
QD score: 24.16 Coverage: 33.25                                         
QD score: 24.34 Coverage: 33.5                                          
QD score: 24.53 Coverage: 33.75                                         
QD score: 24.71 Coverage: 34.0                                          
QD score: 24.9 Coverage: 34.25                                          
QD score: 25.09 Coverage: 34.5                                          
QD score: 25.43 Coverage: 35.0                                          
QD score: 25.79 Coverage: 35.5                                          
QD score: 26.16 Coverage: 36.0                                          
QD score: 26.53 Coverage: 36.5                                          
QD score: 26.92 Coverage: 37.0                                          
QD score: 27.27 Coverage: 37.5                                          
QD score: 27.83 Coverage: 38.25                                         
QD score: 28.01 Coverage: 38.5                                          
QD score: 28.38 Coverage: 39.0                                          
QD score: 28.57 Coverage: 39.25                                         
QD score: 28.93 Coverage: 39.75                                         
QD score: 29.3 Coverage: 40.25                                          
QD score: 29.66 Coverage: 40.75                                         
QD score: 30.03 Coverage: 41.25                                         
QD score: 30.39 Coverage: 41.75                                         
QD score: 30.4 Coverage: 41.75                                          
QD score: 30.4 Coverage: 41.75                                          
QD score: 30.57 Coverage: 42.0                                          
QD score: 30.92 Coverage: 42.5                                          
QD score: 31.28 Coverage: 43.0                                          
QD score: 31.65 Coverage: 43.5                                          
QD score: 32.19 Coverage: 44.25                                         
QD score: 32.37 Coverage: 44.5                                          
QD score: 32.56 Coverage: 44.75                                         
QD score: 32.74 Coverage: 45.0                                          
QD score: 32.92 Coverage: 45.25                                         
QD score: 33.1 Coverage: 45.5                                           
QD score: 33.11 Coverage: 45.5                                          
QD score: 33.29 Coverage: 45.75                                         
QD score: 33.64 Coverage: 46.25                                         
QD score: 33.82 Coverage: 46.5                                          
QD score: 33.83 Coverage: 46.5                                          
QD score: 33.83 Coverage: 46.5                                          
QD score: 33.83 Coverage: 46.5                                          
QD score: 34.01 Coverage: 46.75                                         
QD score: 34.2 Coverage: 47.0                                           
QD score: 34.56 Coverage: 47.5                                          
QD score: 34.56 Coverage: 47.5                                          
QD score: 34.75 Coverage: 47.75                                         
QD score: 34.75 Coverage: 47.75                                         
QD score: 35.11 Coverage: 48.25                                         
QD score: 35.28 Coverage: 48.5                                          
QD score: 35.46 Coverage: 48.75                                         
QD score: 35.65 Coverage: 49.0                                          
QD score: 35.84 Coverage: 49.25                                         
QD score: 36.02 Coverage: 49.5                                          
QD score: 36.22 Coverage: 49.75                                         
QD score: 36.4 Coverage: 50.0                                           
QD score: 36.42 Coverage: 50.0                                          
QD score: 36.79 Coverage: 50.5                                          
QD score: 36.97 Coverage: 50.74999999999999                             
QD score: 36.97 Coverage: 50.74999999999999                             
QD score: 37.15 Coverage: 51.0                                          
QD score: 37.15 Coverage: 51.0                                          
QD score: 37.33 Coverage: 51.24999999999999                             
QD score: 37.69 Coverage: 51.74999999999999                             
QD score: 37.69 Coverage: 51.74999999999999                             
QD score: 37.69 Coverage: 51.74999999999999                             
QD score: 37.88 Coverage: 52.0                                          
QD score: 37.89 Coverage: 52.0                                          
QD score: 38.07 Coverage: 52.25                                         
QD score: 38.07 Coverage: 52.25                                         
QD score: 38.44 Coverage: 52.75                                         
QD score: 38.45 Coverage: 52.75                                         
QD score: 38.63 Coverage: 53.0                                          
QD score: 38.65 Coverage: 53.0                                          
QD score: 38.67 Coverage: 53.0                                          
QD score: 38.85 Coverage: 53.25                                         
QD score: 39.04 Coverage: 53.5                                          
QD score: 39.04 Coverage: 53.5                                          
QD score: 39.21 Coverage: 53.75                                         
QD score: 39.4 Coverage: 54.0                                           
QD score: 39.4 Coverage: 54.0                                           
QD score: 39.76 Coverage: 54.50000000000001                             
QD score: 40.11 Coverage: 55.00000000000001                             
QD score: 40.11 Coverage: 55.00000000000001                             
QD score: 40.11 Coverage: 55.00000000000001                             
QD score: 40.3 Coverage: 55.25                                          
QD score: 40.3 Coverage: 55.25                                          
QD score: 40.3 Coverage: 55.25                                          
QD score: 40.48 Coverage: 55.50000000000001                             
QD score: 40.82 Coverage: 56.00000000000001                             
QD score: 41.0 Coverage: 56.25                                          
QD score: 41.17 Coverage: 56.49999999999999                             
QD score: 41.35 Coverage: 56.75                                         
QD score: 41.36 Coverage: 56.75                                         
QD score: 41.73 Coverage: 57.25                                         
QD score: 41.73 Coverage: 57.25                                         
QD score: 41.74 Coverage: 57.25                                         
QD score: 41.93 Coverage: 57.49999999999999                             
Iterations: 100%|█████████████████████| 200/200 [14:21<00:00,  4.31s/it]

```

##### Cell 27 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```text
## Visualization
### Visualizing the Archive
We use the `grid_archive_heatmap` function to visualize the grid archive as a heatmap.
```

##### Cell 28 (code)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```python
from matplotlib import pyplot as plt

from ribs.visualize import grid_archive_heatmap

plt.figure(figsize=(6, 4.5))
grid_archive_heatmap(archive, aspect="equal", vmin=0, vmax=100)
plt.xlabel("Diversity Metric 1")
plt.ylabel("Diversity Metric 2")
```

Saved output: [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```text
Text(0, 0.5, 'Diversity Metric 2')
```

Saved output: [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```text
<Figure size 600x450 with 2 Axes>
```

##### Cell 29 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```text
We can see that the archive is filled with diverse images, and the diversity metrics are well distributed.
```

##### Cell 30 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```text
### Visualizing the Results

Now let's view the images from across the archive generated by QDHF. We randomly sample images in the grid with measures spread across the diversity dimensions.
```

##### Cell 31 (code)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```python
import itertools

# Modify this to determine how many images to plot along each dimension.
img_freq = (
    4,  # Number of columns of images.
    4,  # Number of rows of images.
)

# List of images.
imgs = []

# Convert archive to a df with solutions available.
df = archive.data(return_type="pandas")

# Compute the min and max measures for which solutions were found.
measure_bounds = np.array(
    [
        (df["measures_0"].min(), df["measures_0"].max()),
        (df["measures_1"].min(), df["measures_1"].max()),
    ]
)

archive_bounds = np.array(
    [archive.boundaries[0][[0, -1]], archive.boundaries[1][[0, -1]]]
)


delta_measures_0 = (archive_bounds[0][1] - archive_bounds[0][0]) / img_freq[0]
delta_measures_1 = (archive_bounds[1][1] - archive_bounds[1][0]) / img_freq[1]


for col, row in itertools.product(range(img_freq[1]), range(img_freq[0])):
    # Compute bounds of a box in measure space.
    measures_0_low = archive_bounds[0][0] + delta_measures_0 * row
    measures_0_high = archive_bounds[0][0] + delta_measures_0 * (row + 1)
    measures_1_low = archive_bounds[1][0] + delta_measures_1 * col
    measures_1_high = archive_bounds[1][0] + delta_measures_1 * (col + 1)

    if row == 0:
        measures_0_low = measure_bounds[0][0]
    if col == 0:
        measures_1_low = measure_bounds[1][0]
    if row == img_freq[0] - 1:
        measures_0_high = measure_bounds[0][1]
    if col == img_freq[1] - 1:
        measures_0_high = measure_bounds[1][1]

    # Query for a solution with measures within this box.
    query_string = (
        f"{measures_0_low} <= measures_0 & measures_0 <= {measures_0_high} & "
        f"{measures_1_low} <= measures_1 & measures_1 <= {measures_1_high}"
    )
    df_box = df.query(query_string)

    if not df_box.empty:
        # Randomly sample a solution from the box.
        # Stable Diffusion solutions have SD_IN_CHANNELS * SD_IN_HEIGHT * SD_IN_WIDTH
        # dimensions, so the final solution col is solution_(x-1).
        sol = (
            df_box.loc[
                :,
                "solution_0" : f"solution_{SD_IN_CHANNELS * SD_IN_HEIGHT * SD_IN_WIDTH - 1}",
            ]
            .sample(n=1)
            .iloc[0]
        )

        # Convert the latent vector solution to an image.
        latents = torch.tensor(sol.to_numpy()).reshape(
            (1, SD_IN_CHANNELS, SD_IN_HEIGHT, SD_IN_WIDTH)
        )
        latents = latents.to(TORCH_DTYPE).to(DEVICE)
        img = SDPIPE(
            prompt,
            num_images_per_prompt=1,
            latents=latents,
            # num_inference_steps=1,  # For testing.
        ).images[0]

        img = torch.from_numpy(np.array(img)).permute(2, 0, 1) / 255.0
        imgs.append(img)
    else:
        imgs.append(torch.zeros((3, IMG_HEIGHT, IMG_WIDTH)))
```

##### Cell 32 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```text
And the images are plotted below as a single collage.
```

##### Cell 33 (code)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```python
from torchvision.utils import make_grid


def create_archive_tick_labels(measure_range, num_ticks):
    delta = (measure_range[1] - measure_range[0]) / num_ticks
    ticklabels = [round(delta * p + measure_range[0], 3) for p in range(num_ticks + 1)]
    return ticklabels


plt.figure(figsize=(img_freq[0] * 2, img_freq[0] * 2))
img_grid = make_grid(imgs, nrow=img_freq[0], padding=0)
img_grid = np.transpose(img_grid.cpu().numpy(), (1, 2, 0))
plt.imshow(img_grid)

plt.xlabel("")
num_x_ticks = img_freq[0]
x_ticklabels = create_archive_tick_labels(measure_bounds[0], num_x_ticks)
x_tick_range = img_grid.shape[1]
x_ticks = np.arange(0, x_tick_range + 1e-9, step=x_tick_range / num_x_ticks)
plt.xticks(x_ticks, x_ticklabels)

plt.ylabel("")
num_y_ticks = img_freq[1]
y_ticklabels = create_archive_tick_labels(measure_bounds[1], num_y_ticks)
y_ticklabels.reverse()
y_tick_range = img_grid.shape[0]
y_ticks = np.arange(0, y_tick_range + 1e-9, step=y_tick_range / num_y_ticks)
plt.yticks(y_ticks, y_ticklabels)

plt.tight_layout()
```

Saved output: [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```text
<Figure size 800x800 with 1 Axes>
```

##### Cell 34 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```text
We can see that QDHF generates a diverse collection of images with some visible trends of diversity such as object size (smaller on upper left, larger on bottom right), scene layout (dark sky on the top, dusty sky on the bottom), etc.


*This tutorial uses light-weight models and smaller batch sizes for faster inference. You should get better results using the [official implementation of QDHF](https://github.com/ld-ing/qdhf).*
```

##### Cell 35 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```text
## Conclusion

In this tutorial, we explored using the QDHF algorithm to generate diverse images given the text prompt. We showed how QDHF can be implemented using pyribs, and how we use it in a pipeline with Stable Diffusion. The generated images show variations and visible trends of diversity without the need to manually specify diversity metrics.

```

##### Cell 36 (markdown)

[Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

```text
## Citation

If you find this tutorial useful, please cite our paper:

```text
@inproceedings{ding2024qdhf,
  author = {Ding, Li and Zhang, Jenny and Clune, Jeff and Spector, Lee and Lehman, Joel},
  title = {Quality diversity through human feedback: towards open-ended diversity-driven optimization},
  year = {2024},
  publisher = {JMLR.org},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  articleno = {441},
  numpages = {19},
  location = {Vienna, Austria},
  series = {ICML'24}
}
```
```

### Inferences
- No metric interpolation was applied. Image iteration numbers are assigned by the order of the 200 sequential log lines in the 1–200 loop. [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb)

### Gaps
- Embedded binary figures are not reproduced as new files; their plotting code and textual interpretation are included.

## Papers and provenance

### Takeaway
The tutorials are small illustrative variations, rather than exact paper-scale reproductions. [Stories](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb), [images](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdhf.ipynb).

### Cited Findings
- **Quality-Diversity through AI Feedback**. Herbie Bradley, Andrew Dai, Hannah Teufel, Jenny Zhang, Koen Oostermeijer, Marco Bellagente, Jeff Clune, Kenneth Stanley, Grégory Schott, Joel Lehman. ICLR 2024; arXiv:2310.13032, first submitted 2023. [Conference paper](https://openreview.net/pdf?id=owokKCrGYr), [arXiv](https://arxiv.org/abs/2310.13032). Headline results: greater coverage with high-quality creative text than non-QD controls; human evaluations support agreement with AI ratings and prefer the quality/diversity of QDAIF sets. [Author project](https://qdaif.github.io/).
- **Quality Diversity through Human Feedback: Towards Open-Ended Diversity-Driven Optimization**. Li Ding, Jenny Zhang, Jeff Clune, Lee Spector, Joel Lehman. ICML 2024, PMLR 235:11072–11090; arXiv:2310.12103. [Proceedings](https://proceedings.mlr.press/v235/ding24h.html), [arXiv](https://arxiv.org/abs/2310.12103). Headline results: learned diversity metrics outperform automatic discovery baselines and match hand-designed metrics in robotics/reinforcement-learning benchmarks; diffusion image sets improve diversity and receive more favorable user-study responses. [Proceedings abstract](https://proceedings.mlr.press/v235/ding24h.html).
- Built-in GaussianEmitter implementation samples parents then adds normal noise; full ask implementation inspected at the release tag. [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/ribs/emitters/_gaussian_emitter.py).

### Inferences
- The story tutorial deliberately substitutes averaged 1–10 ratings and directional mutation for the paper’s predominant logit scoring and Language Model Crossover. [Source](https://github.com/icaros-usc/pyribs/blob/v0.12.1/tutorials/qdaif.ipynb).

### Gaps
- Numerical paper headline effect sizes: not confirmed from inspected primary-source abstracts; not found.
