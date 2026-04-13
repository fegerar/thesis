"""Shape-graph annotation for DFL (Bassek et al.) tracking data.

Re-implements the Brandes et al. 2025 shape-graph algorithm from scratch on top
of `scipy.spatial.Delaunay` only — no `shapegraph` package is used.
"""

from .shape_graph import build_shape_graph, infer_team_roles, infer_joint_zone

__all__ = ["build_shape_graph", "infer_team_roles", "infer_joint_zone"]
