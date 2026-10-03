"""Measurement machinery, kept separate from the pricing models.

``pricing`` answers "what is this option worth?". ``experiments`` answers "how
well does that method work, and what does it cost?" -- which is the actual
research question of this project.

- ``harness`` runs an estimator many times over many seeds and reports how
  accurate, how honest and how fast it was. Phase 4's variance-reduced
  estimators plug into the same harness unchanged.
"""
