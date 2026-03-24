"""
Shapegraph tokenizer pipeline.

Converts raw Bassek et al. (2025) tracking data into VQ-VAE token sequences
for autoregressive modeling of tactical soccer states.
"""

from .extract import load_shapegraphs_per_match
from .deduplicate import deduplicate_match
from .encode import VQVAEEncoder
from .tokenize import build_token_sequence
from .pipeline import run_pipeline
