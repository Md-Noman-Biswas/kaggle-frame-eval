"""
Public API for the eval package.

We import only the lightweight metrics module at package load. The runner
is imported lazily to keep cold-start fast and avoid pulling tqdm/transformers
when callers only need metrics.
"""

from .metrics import compute_metrics, format_metrics, duration_bucket


def run_evaluation(*args, **kwargs):
    """Lazy proxy: imports the runner only when first called."""
    from .runner import run_evaluation as _impl
    return _impl(*args, **kwargs)


__all__ = ["run_evaluation", "compute_metrics", "format_metrics", "duration_bucket"]
