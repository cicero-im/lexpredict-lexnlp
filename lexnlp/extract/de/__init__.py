"""German (de) extraction support for LexNLP.

Surfaces the German identifier extractors (Steuer-IdNr, USt-IdNr,
Handelsregisternummer) ported from :mod:`lexnlp.extract.pt.identifiers`.
The existing per-feature extractors (``amounts``, ``citations``,
``courts``, ``dates``, ``definitions``, …) remain importable directly
from their submodules.
"""

__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


from lexnlp.extract.de.identifiers import (
    DeIdentifierMatch,
    get_hrb_annotations,
    get_identifier_annotations,
    get_steuer_idnr_annotations,
    get_ust_idnr_annotations,
)

__all__ = [
    "DeIdentifierMatch",
    "get_hrb_annotations",
    "get_identifier_annotations",
    "get_steuer_idnr_annotations",
    "get_ust_idnr_annotations",
]
