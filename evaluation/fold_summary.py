import math
import statistics


def _finite(values):
    return [
        float(v)
        for v in values
        if v is not None and math.isfinite(float(v))
    ]


def summarize_folds(rows, metrics):
    """Summarize metric dictionaries using all supplied folds.

    Standard deviation is the sample SD (equivalent to ``ddof=1``).
    """
    out = {}

    for metric in metrics:
        vals = _finite(
            row.get(metric)
            for row in rows
        )

        out[metric + "_n"] = len(vals)

        if not vals:
            out[metric + "_mean"] = None
            out[metric + "_sd"] = None
        elif len(vals) == 1:
            out[metric + "_mean"] = vals[0]
            out[metric + "_sd"] = None
        else:
            out[metric + "_mean"] = statistics.mean(vals)
            out[metric + "_sd"] = statistics.stdev(vals)

    return out
