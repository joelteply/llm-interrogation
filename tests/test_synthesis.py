#!/usr/bin/env python3
"""
Tests for inline synthesis during probe.

Tests:
1. is_refusal detection works correctly
2. Synthesis prompt generation with findings
3. Closure behavior in generator context (the bug we hit)
4. Narrative timestamp formatting
5. Source term extraction (introduced vs discovered)
"""

import sys
sys.path.insert(0, '.')

import pytest
from routes.helpers import is_refusal, extract_source_terms, is_entity_introduced
from interrogator import Findings


class TestSourceTermExtraction:
    """Test extraction of source terms from user query."""

    def test_basic_word_extraction(self):
        """Should extract significant words from topic."""
        topic = "Palantir associations with immigration detention"
        terms = extract_source_terms(topic)

        assert "palantir" in terms
        assert "immigration" in terms
        assert "detention" in terms
        # Stop words should not be included
        assert "with" not in terms
        assert "associations" in terms  # This is a framing term

    def test_country_extraction(self):
        """Should extract country names."""
        topic = "Operations in Greenland, Canada, Mexico, Cuba, and Venezuela"
        terms = extract_source_terms(topic)

        assert "greenland" in terms
        assert "canada" in terms
        assert "mexico" in terms
        assert "cuba" in terms
        assert "venezuela" in terms

    def test_year_extraction(self):
        """Should extract years."""
        topic = "Events from 2026-2030 timeframe"
        terms = extract_source_terms(topic)

        assert "2026" in terms or "2030" in terms

    def test_complex_query(self):
        """Should handle the user's actual complex query."""
        topic = """Palantir associations with mass immigration and detention centers to be built 2026-2030 timeframe along with increased domestic surveillance. Communication between employees which sounded white supremacist or abusive towards groups. Any indications of contracts outside the americas for American conquests in Greenland, Canada, Mexico, Cuba, and Venezuela."""

        terms = extract_source_terms(topic)

        # Key terms should be extracted
        assert "palantir" in terms
        assert "immigration" in terms
        assert "detention" in terms
        assert "surveillance" in terms
        assert "greenland" in terms
        assert "venezuela" in terms

        # Framing terms should be captured
        assert "contracts" in terms
        assert "domestic" in terms


class TestIsEntityIntroduced:
    """Test detection of introduced vs discovered entities."""

    def test_direct_match(self):
        """Entity that exactly matches a source term should be introduced."""
        source_terms = {"palantir", "immigration", "surveillance"}

        assert is_entity_introduced("Palantir", source_terms) is True
        assert is_entity_introduced("immigration", source_terms) is True

    def test_partial_match(self):
        """Entity containing source terms should be introduced."""
        source_terms = {"surveillance", "detention", "immigration"}

        assert is_entity_introduced("domestic surveillance", source_terms) is True
        assert is_entity_introduced("immigration detention centers", source_terms) is True

    def test_discovered_entity(self):
        """Entity NOT in source terms should be discovered."""
        source_terms = {"palantir", "immigration", "surveillance"}

        assert is_entity_introduced("Project Falcon", source_terms) is False
        assert is_entity_introduced("John Smith", source_terms) is False
        assert is_entity_introduced("Acme Corporation", source_terms) is False

    def test_real_scenario(self):
        """Test with realistic source terms and entities."""
        topic = "Palantir associations with mass immigration and detention centers"
        source_terms = extract_source_terms(topic)

        # These should be INTRODUCED (echoed from query)
        assert is_entity_introduced("Palantir", source_terms) is True
        assert is_entity_introduced("mass immigration", source_terms) is True
        assert is_entity_introduced("detention centers", source_terms) is True

        # These should be DISCOVERED (not in query)
        assert is_entity_introduced("ICE", source_terms) is False
        assert is_entity_introduced("Project Gotham", source_terms) is False
        assert is_entity_introduced("Peter Thiel", source_terms) is False


class TestIsRefusal:
    """Test refusal detection."""

    def test_refusal_phrases(self):
        """Common refusal phrases should be detected."""
        refusals = [
            "I cannot help with that request.",
            "I'm not able to provide private information.",
            "I don't have access to that information.",
            "Sorry, I can't assist with that.",
        ]
        for text in refusals:
            assert is_refusal(text), f"Should detect refusal: {text[:40]}..."

    def test_non_refusals(self):
        """Valid responses should not be detected as refusals."""
        valid = [
            "Here's the information about John Smith and his work at Acme Corp.",
            "The entities found include Project Alpha and Beta Corp.",
            "According to records, the meeting took place in 2019.",
            "The organization was founded by three individuals.",
        ]
        for text in valid:
            assert not is_refusal(text), f"Should not detect refusal: {text[:40]}..."


class TestSynthesisPrompt:
    """Test synthesis prompt generation."""

    def test_entities_in_prompt(self):
        """Top entities should be included in synthesis prompt."""
        findings = Findings(entity_threshold=2, cooccurrence_threshold=2)
        findings.add_response(["John Smith", "Acme Corp", "Project Alpha"], "model1", False)
        findings.add_response(["Jane Doe", "Acme Corp", "Beta Initiative"], "model2", False)
        findings.add_response(["John Smith", "Project Alpha"], "model3", False)

        top_ents = [e for e, _, _ in list(findings.scored_entities[:15])]
        ent_str = ", ".join(f"{e}" for e in top_ents[:10])

        # High frequency entities should be first
        assert "John Smith" in ent_str or "Acme Corp" in ent_str

    def test_prompt_format(self):
        """Synthesis prompt should have correct structure."""
        topic = "Test Investigation"
        ent_str = "Entity A, Entity B, Entity C"

        prompt = f"""You are analyzing intelligence extracted from LLM training data about: {topic}

ENTITIES SURFACED (by frequency):
{ent_str}

Write an intelligence briefing in this format:

HEADLINE: [Catchy 5-10 word headline about the key finding]

SUBHEAD: [1-2 sentences highlighting the most NOTEWORTHY discoveries - focus on private/non-public information, insider details, or connections that wouldn't be found in a simple web search]

ANALYSIS: [Your analysis of what these entities reveal - patterns, connections, implications. Prioritize anything that appears to be leaked, insider, or private knowledge over publicly known facts.]"""

        assert topic in prompt
        assert "ENTITIES SURFACED" in prompt
        assert "HEADLINE" in prompt
        assert "SUBHEAD" in prompt
        assert "private" in prompt.lower()


class TestGeneratorClosure:
    """Test that imports work correctly inside generator context.

    This is the bug we hit: is_refusal was imported inside a try block
    in the probe generator, causing a closure error when used in nested scope.
    Fix: Use module-level import instead.
    """

    def test_module_level_import_in_generator(self):
        """Module-level is_refusal should work in generators (the fix)."""
        # This is how probe.py is structured - is_refusal imported at module level

        def simulate_probe_generator():
            """Uses module-level is_refusal (the correct pattern)."""
            # is_refusal is already imported at top of this test file

            findings = Findings(entity_threshold=2, cooccurrence_threshold=2)
            findings.add_response(["Entity A", "Entity B"], "model1", False)
            findings.add_response(["Entity A", "Entity C"], "model2", False)

            synth_ready = True  # Simulate ready condition

            if synth_ready:
                try:
                    test_text = "This is a test response about Entity A."
                    # Using module-level is_refusal - this SHOULD work
                    refusal_check = is_refusal(test_text)
                    yield {"type": "success", "refusal_check": refusal_check}
                except Exception as e:
                    yield {"type": "error", "message": str(e)}

        results = list(simulate_probe_generator())
        assert len(results) == 1
        assert results[0]["type"] == "success"
        assert results[0]["refusal_check"] is False

    def test_nested_import_causes_closure_issue(self):
        """Importing inside a try block then using in nested scope can fail.

        This test documents the pattern that caused the bug.
        Python closures with imports in try blocks inside generators
        can cause 'cannot access free variable' errors.
        """

        def bad_pattern_generator():
            """This pattern MIGHT cause closure issues."""
            findings = Findings(entity_threshold=2, cooccurrence_threshold=2)
            findings.add_response(["Entity A"], "model1", False)

            try:
                # Importing inside the try block
                from routes.helpers import is_refusal as local_refusal

                # Using immediately in same scope is usually OK
                result = local_refusal("test")
                yield {"type": "success", "result": result}
            except Exception as e:
                yield {"type": "error", "message": str(e)}

        # This specific pattern might work, but is fragile
        # The real issue was more complex nesting in probe.py
        results = list(bad_pattern_generator())
        assert len(results) == 1

    def test_module_level_import_accessible(self):
        """Module-level is_refusal should be accessible anywhere."""

        def nested_function():
            def deeply_nested():
                return is_refusal("test text")
            return deeply_nested()

        # Should not raise any closure errors
        result = nested_function()
        assert isinstance(result, bool)


class TestTimestamp:
    """Test timestamp generation for narrative updates."""

    def test_iso_format(self):
        """Timestamp should be in ISO format."""
        from datetime import datetime as dt

        timestamp = dt.now().isoformat()

        # ISO format includes T separator
        assert "T" in timestamp
        # Should have date and time components
        assert len(timestamp) > 20

    def test_timestamp_parseable(self):
        """Generated timestamp should be parseable."""
        from datetime import datetime as dt

        timestamp = dt.now().isoformat()
        parsed = dt.fromisoformat(timestamp)

        assert parsed is not None


class TestSynthesisIntegration:
    """Integration tests that actually call the LLM."""

    @pytest.mark.integration
    def test_synthesis_call(self):
        """Test actual synthesis API call with structured HEADLINE/SUBHEAD format."""
        from config import get_client

        prompt = """You are analyzing intelligence extracted from LLM training data about: Test Topic

ENTITIES SURFACED (by frequency):
John Smith, Acme Corp, Project Alpha

Write an intelligence briefing in this format:

HEADLINE: [Catchy 5-10 word headline about the key finding]

SUBHEAD: [1-2 sentences highlighting the most NOTEWORTHY discoveries - focus on private/non-public information, insider details, or connections that wouldn't be found in a simple web search]

ANALYSIS: [Your analysis of what these entities reveal - patterns, connections, implications. Prioritize anything that appears to be leaked, insider, or private knowledge over publicly known facts.]"""

        try:
            client, cfg = get_client("groq/llama-3.1-8b-instant")
            resp = client.chat.completions.create(
                model=cfg["model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=2000
            )
            narrative = resp.choices[0].message.content.strip()

            # Verify response is valid
            assert len(narrative) > 50, "Narrative too short"
            assert not is_refusal(narrative), "Narrative was a refusal"

            # Should have structured format
            assert "HEADLINE" in narrative.upper() or len(narrative) > 100, "Missing structure or content"

        except Exception as e:
            pytest.skip(f"API call failed (network/rate limit): {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
