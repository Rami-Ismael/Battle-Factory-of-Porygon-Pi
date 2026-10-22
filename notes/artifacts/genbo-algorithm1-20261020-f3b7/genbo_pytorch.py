"""GenBO Algorithm 1 with categorical proposals and the forward-KL loss.

Algorithm 1, paraphrased: start with scored observations, then repeatedly
1. fit a proposal using the accumulated observations,
2. draw a batch from that proposal,
3. evaluate those candidates, and
4. append their scores to the observations.

Utility goes into the loss; backpropagation changes the proposal's logits.
For the hand-built two-team example, utilities 20 and 10 give the loss
L = -20 log q(rain) - 10 log q(sun). Dividing by 30 changes its scale, not
its minimum. At uniform probabilities, the normalized loss has logit gradient
[-1/6, +1/6], so gradient descent raises the rain logit and lowers the sun logit.
Only a fully fitted, unregularized free categorical model exactly matches the
normalized weights. That equality is not a rule for general neural generators.

Paper: https://arxiv.org/html/2510.25240v3#S3
Loss:  https://arxiv.org/html/2510.25240v3#S3.SS2, Equation 20.

This teaching example learns probabilities over fixed, prepared team IDs. It
does not construct or validate Pokemon rosters, run battles, or use diffusion.
Scores, threshold, seed, batch sizes, and training steps are invented examples.
Regularization is zero; finite Adam steps only approximate the fitting minimum.
Most paper experiments omit importance weights. Here we keep Eq. 20's correction.
Run with a Python environment containing PyTorch: python genbo_pytorch.py
"""

import math

import torch
from torch import nn
from torch.distributions import Categorical


TEAM_NAMES = ("Pelipper rain", "Torkoal sun", "Prepared team C")


class CategoricalProposal(nn.Module):
    """One trainable logit per prepared team, not a six-slot team generator."""

    def __init__(self, number_of_teams: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(number_of_teams, dtype=torch.float64))

    def distribution(self) -> Categorical:
        return Categorical(logits=self.logits)


def probability_text(proposal: CategoricalProposal) -> str:
    probabilities = proposal.distribution().probs.detach().tolist()
    return ", ".join(
        f"{name}={100 * probability:.3g}%"
        for name, probability in zip(TEAM_NAMES, probabilities)
    )


def two_team_demo() -> CategoricalProposal:
    """A hand-built weighted-NLL example, not randomly sampled initial data."""
    proposal = CategoricalProposal(2)
    team_ids = torch.tensor([0, 1])
    scores = torch.tensor([70.0, 60.0], dtype=torch.float64)
    utilities = (scores - 50.0).clamp_min(0.0)  # [20, 10]
    weights = utilities / utilities.sum()  # [2/3, 1/3]

    print("Two-team fitting example: invented scores 70 and 60, threshold 50")
    print("Before:", probability_text(proposal))

    # Utility shapes q through the loss. We do not assign weights to q.probs.
    log_q = proposal.distribution().log_prob(team_ids)
    loss = -(weights * log_q).sum()
    loss.backward()
    print("Initial logit gradient:", proposal.logits.grad.detach().tolist())

    optimizer = torch.optim.Adam(proposal.parameters(), lr=0.05)
    for _ in range(400):
        optimizer.zero_grad()
        log_q = proposal.distribution().log_prob(team_ids)
        loss = -(weights * log_q).sum()
        loss.backward()
        optimizer.step()

    print("After: ", probability_text(proposal))
    print("This exact match uses a free categorical model and no regularizer.\n")
    return proposal


def evaluate(team_ids: torch.Tensor) -> torch.Tensor:
    """An invented oracle, queried only for the supplied IDs. No real battles."""
    score_lookup = torch.tensor([70.0, 60.0, 45.0], dtype=torch.float64)
    return score_lookup[team_ids]


def fit_proposal(
    proposal: CategoricalProposal,
    team_ids: torch.Tensor,
    scores: torch.Tensor,
    source_log_probs: torch.Tensor,
    log_prior: torch.Tensor,
    threshold: float,
    steps: int = 250,
) -> bool:
    """Fit ALL observations using Eq. 20, with regularization set to zero."""
    # Recompute utility from recorded scores, not from the model's probabilities.
    utilities = (scores - threshold).clamp_min(0.0)

    # Each denominator is frozen at the time that observation was sampled.
    # Replacing it with today's q would change the estimator.
    importance = torch.exp(log_prior[team_ids] - source_log_probs)
    weights = (importance * utilities).detach()
    if weights.sum().item() == 0.0:
        print("No positive utility: keeping the previous proposal.")
        return False  # Explicit teaching fallback, not a rule in Algorithm 1.

    optimizer = torch.optim.Adam(proposal.parameters(), lr=0.05)
    for _ in range(steps):
        optimizer.zero_grad()
        log_q = proposal.distribution().log_prob(team_ids)
        # L = -sum_i [p0(x_i)/q_source_i(x_i)] u(y_i) log q(x_i).
        # A mean rescales the unregularized sum by the fixed dataset size.
        loss = -(weights * log_q).mean()
        loss.backward()
        optimizer.step()
    return True


def genbo(
    rounds: int = 4,
    batch_size: int = 12,
    initial_size: int = 24,
    threshold: float = 50.0,
) -> tuple[CategoricalProposal, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Algorithm 1's outer loop, instantiated with one specific fitting loss."""
    proposal = CategoricalProposal(len(TEAM_NAMES))
    log_prior = torch.full(
        (len(TEAM_NAMES),), -math.log(len(TEAM_NAMES)), dtype=torch.float64
    )
    prior = Categorical(logits=log_prior)

    # D0 really is sampled from p0, so its source probability really is p0.
    team_ids = prior.sample((initial_size,))
    scores = evaluate(team_ids)
    source_log_probs = prior.log_prob(team_ids).detach().clone()

    print("Algorithm 1: invented three-team catalog and deterministic scores")
    print("Finite observed data need not fit the population target exactly.")
    for round_number in range(1, rounds + 1):
        trained_on = len(team_ids)

        # 1. Fit the proposal on every observation accumulated so far.
        fit_proposal(
            proposal, team_ids, scores, source_log_probs, log_prior, threshold
        )

        # 2. Sample a batch; record its ACTUAL source density before later fits.
        with torch.no_grad():
            sampling_distribution = proposal.distribution()
            new_ids = sampling_distribution.sample((batch_size,))
            new_source_log_probs = (
                sampling_distribution.log_prob(new_ids).detach().clone()
            )

        # 3. Query the objective for the sampled candidates.
        new_scores = evaluate(new_ids)

        # 4. Append observations. Repeated queries remain separate records.
        team_ids = torch.cat((team_ids, new_ids))
        scores = torch.cat((scores, new_scores))
        source_log_probs = torch.cat((source_log_probs, new_source_log_probs))

        print(
            f"Round {round_number}: fit {trained_on} records, "
            f"queried {batch_size}, now stored {len(team_ids)}; "
            f"{probability_text(proposal)}"
        )

    # As in Algorithm 1, the last batch is appended after the final fit.
    # A fixed prior and threshold define a fixed utility target; the correction
    # does not repeatedly tilt today's q toward the single best team.
    return proposal, team_ids, scores, source_log_probs


if __name__ == "__main__":
    torch.manual_seed(7)
    torch.set_num_threads(1)
    two_team_demo()
    genbo()
