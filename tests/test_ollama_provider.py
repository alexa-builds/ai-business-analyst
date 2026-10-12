"""Offline tests for the optional Ollama adapter and safe interpretation gate."""

import unittest
from unittest.mock import patch

import pandas as pd

from business_analyst import answer_business_question
from ollama_provider import (
    OllamaUnavailable,
    UnsafeInterpretation,
    generate_interpretation,
    interpretation_context,
    list_local_models,
    validate_interpretation,
)


class OllamaProviderTests(unittest.TestCase):
    def test_local_service_unavailable_does_not_break_python_answers(self):
        with patch("ollama_provider.http.client.HTTPConnection", side_effect=OSError("not running")):
            with self.assertRaises(OllamaUnavailable):
                list_local_models()

        answer = answer_business_question(
            pd.DataFrame({"value": [2, 6]}),
            "What is the average value?",
        )
        self.assertEqual(answer["facts"][0]["value"], "4.00")

    def test_cloud_tagged_models_are_not_offered(self):
        with patch("ollama_provider._local_request", return_value={
            "models": [{"name": "qwen:latest"}, {"name": "remote:cloud"}, {"name": "gpt-oss:120b-cloud"}]
        }):
            self.assertEqual(list_local_models(), ["qwen:latest"])

    def test_model_context_contains_only_compact_calculated_evidence(self):
        context = interpretation_context({
            "question": "What is the total?",
            "headline": "Total value is 8.00.",
            "facts": [{"label": "Total", "value": "8.00"}],
            "calculation": "Sum of valid values.",
            "limitations": ["Filtered rows only."],
            "raw_rows": [{"private_note": "must not be sent"}],
        })

        self.assertNotIn("raw_rows", context)
        self.assertNotIn("private_note", str(context))
        self.assertEqual(set(context), {"question", "python_finding", "supporting_calculations", "calculation_method", "limitations"})

    def test_nonnumeric_qualitative_response_is_accepted(self):
        self.assertEqual(
            validate_interpretation("The leading group stands apart from the rest of the observed results."),
            "The leading group stands apart from the rest of the observed results.",
        )

    def test_fabricated_numbers_and_causal_claims_are_rejected(self):
        with self.assertRaises(UnsafeInterpretation):
            validate_interpretation("Revenue improved by 27 percent.")
        with self.assertRaises(UnsafeInterpretation):
            validate_interpretation("The campaign caused the higher result.")
        with self.assertRaises(UnsafeInterpretation):
            validate_interpretation("The higher result happened because of stronger demand.")

    def test_interpretation_uses_only_verified_local_model_and_minimal_chat_context(self):
        result = {
            "question": "Which group is highest?",
            "headline": "North is highest at 8.00.",
            "facts": [{"label": "North", "value": "8.00"}],
            "calculation": "Grouped sum.",
            "limitations": ["No cause inferred."],
        }
        response = {"message": {"content": "The leading group is ahead in the observed comparison."}}
        with patch("ollama_provider.list_local_models", return_value=["qwen:latest"]):
            with patch("ollama_provider._local_request", return_value=response) as request:
                text = generate_interpretation("qwen:latest", result)

        self.assertIn("leading group", text)
        request.assert_called_once()
        method, path, payload = request.call_args.args[:3]
        self.assertEqual(method, "POST")
        self.assertEqual(path, "/api/chat")
        user_context = payload["messages"][1]["content"]
        self.assertNotIn("uploaded_rows", user_context)
        self.assertNotIn("environment", user_context)

    def test_nonlocal_model_is_rejected_before_chat_request(self):
        with patch("ollama_provider.list_local_models", return_value=["qwen:latest"]):
            with patch("ollama_provider._local_request") as request:
                with self.assertRaises(OllamaUnavailable):
                    generate_interpretation("qwen:cloud", {"question": "q"})
        request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
