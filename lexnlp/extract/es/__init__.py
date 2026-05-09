"""Spanish (es) extraction support for LexNLP.

Mirrors the existing locale modules (``lexnlp.extract.de`` /
``lexnlp.extract.pt``). Includes ``identifiers`` (DNI / NIE / NIF / CIF
extractors with check-letter validation) ported from the Portuguese
``lexnlp.extract.pt.identifiers`` design.
"""

__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


from lexnlp.extract.es.identifiers import (
    EsIdentifierMatch,
    get_cif_annotations,
    get_dni_annotations,
    get_identifier_annotations,
    get_nie_annotations,
    get_nif_annotations,
)
from lexnlp.extract.es.regulations import (
    get_regulation_annotation_list,
    get_regulation_annotations,
    get_regulation_list,
    get_regulations,
)

__all__ = [
    "EsIdentifierMatch",
    "get_cif_annotations",
    "get_dni_annotations",
    "get_identifier_annotations",
    "get_nie_annotations",
    "get_nif_annotations",
    "get_regulation_annotation_list",
    "get_regulation_annotations",
    "get_regulation_list",
    "get_regulations",
]
