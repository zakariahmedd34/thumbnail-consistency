"""Flask wrapper around the thumbnail-video consistency pipeline.

The ML core (``src/``) is imported but **not** modified by this package.
The Flask layer is a thin adapter: HTTP <-> ``ConsistencyScorer``.
"""
