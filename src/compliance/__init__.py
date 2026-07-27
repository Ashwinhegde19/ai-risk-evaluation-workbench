"""Compliance mapping package.

Maps evaluation findings onto three regulatory frameworks:
    * EU AI Act        (``eu_ai_act``)
    * NIST AI RMF      (``nist_rmf``)
    * ISO/IEC 42001    (``iso_42001``)

The shared helpers in ``_common`` keep severity ordering and risk-tier
aggregation consistent across all three mappers.
"""
