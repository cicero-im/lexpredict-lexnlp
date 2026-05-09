"""Tests for :mod:`lexnlp.extract.pt.pii`."""

__author__ = "ContraxSuite, LLC; LexPredict, LLC"
__copyright__ = "Copyright 2015-2021, ContraxSuite, LLC"
__license__ = "https://github.com/LexPredict/lexpredict-lexnlp/blob/2.3.0/LICENSE"
__version__ = "2.3.0"
__maintainer__ = "LexPredict, LLC"
__email__ = "support@contraxsuite.com"


from unittest import TestCase

from lexnlp.extract.pt.pii import (
    get_email_list,
    get_phone_annotation_list,
    get_phone_list,
)


class TestPtPhones(TestCase):
    def test_extracts_mobile_with_parens(self):
        text = "Contato: (11) 98765-4321 para mais informações."
        ants = get_phone_annotation_list(text)
        self.assertEqual(1, len(ants))
        self.assertEqual("11987654321", ants[0].phone)
        self.assertEqual("pt", ants[0].locale)

    def test_extracts_landline_8_digits(self):
        text = "Ligar para (21) 1234-5678."
        ants = get_phone_annotation_list(text)
        self.assertEqual(1, len(ants))
        self.assertEqual("2112345678", ants[0].phone)

    def test_extracts_with_country_code(self):
        text = "Whatsapp +55 11 91234-5678 disponível."
        ants = get_phone_annotation_list(text)
        self.assertEqual(1, len(ants))
        self.assertEqual("5511912345678", ants[0].phone)

    def test_extracts_0800(self):
        text = "Atendimento 0800 123 4567."
        ants = get_phone_annotation_list(text)
        self.assertEqual(1, len(ants))
        self.assertEqual("08001234567", ants[0].phone)

    def test_does_not_match_short_number(self):
        text = "Sala 123."
        self.assertEqual([], get_phone_list(text))

    def test_no_phone_in_plain_text(self):
        self.assertEqual([], get_phone_list("Sem telefones aqui."))

    def test_legacy_8_digit_local_without_ddd(self):
        """Pre-2012 8-digit landline without DDD (``1234-5678``)."""
        text = "Telefone do escritório: 1234-5678."
        ants = get_phone_annotation_list(text)
        self.assertEqual(1, len(ants))
        self.assertEqual("12345678", ants[0].phone)

    def test_legacy_7_digit_local_without_ddd(self):
        """Pre-2002 7-digit local number (``234-5678``)."""
        text = "Antigo número: 234-5678."
        ants = get_phone_annotation_list(text)
        self.assertEqual(1, len(ants))
        self.assertEqual("2345678", ants[0].phone)

    def test_oxx_operator_placeholder(self):
        """``0XX11 1234-5678`` — late-1990s telecom-deregulation prefix."""
        text = "Ligar (0XX11) 1234-5678 a partir de outra cidade."
        ants = get_phone_annotation_list(text)
        self.assertEqual(1, len(ants))
        # Digits captured: "0xx" -> stripped, leaving "11" + "12345678".
        # The exact normalisation depends on the regex, but the surface
        # form must be preserved on the annotation.
        self.assertIn("0XX11", ants[0].text)

    def test_explicit_operator_code_prefix(self):
        """``0 32 11 1234-5678`` — explicit carrier code (32 = Embratel)."""
        text = "Discar 032 11 1234-5678 para atendimento interurbano."
        ants = get_phone_annotation_list(text)
        self.assertEqual(1, len(ants))


class TestPtEmails(TestCase):
    def test_extracts_simple_email(self):
        text = "Envie para contato@exemplo.com.br por favor."
        emails = get_email_list(text)
        self.assertEqual(1, len(emails))
        self.assertEqual("contato@exemplo.com.br", emails[0])

    def test_extracts_with_plus(self):
        text = "Use info+legal@empresa.com para o setor jurídico."
        emails = get_email_list(text)
        self.assertEqual(1, len(emails))
        self.assertEqual("info+legal@empresa.com", emails[0])

    def test_extracts_multiple(self):
        text = "Foo a@b.com bar c@d.org baz."
        emails = get_email_list(text)
        self.assertEqual(2, len(emails))

    def test_no_email_in_plain_text(self):
        self.assertEqual([], get_email_list("Sem endereços aqui."))
