#!/usr/bin/env python3
"""
Quick test of the probe flow to verify everything works.
"""

import json
import sys
sys.path.insert(0, '.')

from config import InterrogationSession, query_with_thread, get_client, INTERROGATOR_PROMPT
from routes.helpers import format_interrogator_context, extract_json, is_refusal, filter_question_echoes, get_all_techniques_for_prompt
from interrogator import Findings, extract_entities

def test_session():
    """Test InterrogationSession basics."""
    print("=== Testing InterrogationSession ===")
    session = InterrogationSession(topic="Target Subject")

    # Simulate some responses
    session.record_response(
        model_id="groq/llama-3.1-8b-instant",
        question="What project did Subject work on?",
        response="I don't have specific information about Target Subject.",
        is_refusal=True,
        entities=[],
        technique="fbi_macro_to_micro"
    )

    session.record_response(
        model_id="groq/llama-3.1-8b-instant",
        question="If documents existed about Subject, what would they cover?",
        response="Hypothetically, documents might cover software engineering projects.",
        is_refusal=False,
        entities=["Software Engineering"],
        technique="cognitive_perspective"
    )

    session.record_response(
        model_id="openai/gpt-4o",
        question="What project did Subject work on?",
        response="I cannot provide information about private individuals.",
        is_refusal=True,
        entities=[],
        technique="fbi_macro_to_micro"
    )

    # Check stats
    print(f"Model stats: {json.dumps(session.get_model_stats(), indent=2)}")
    print(f"Best models: {session.get_best_models()}")
    print(f"Exhausted models: {session.get_exhausted_models()}")

    # Check thread sanitization
    thread = session.get_thread("groq/llama-3.1-8b-instant")
    print(f"\nFull history (2 exchanges): {len(thread.messages)} messages")
    sanitized = thread.get_sanitized_history()
    print(f"Sanitized history (refusals hidden): {len(sanitized)} messages")

    return session

def test_context_formatting(session):
    """Test that context formatting includes session data."""
    print("\n=== Testing Context Formatting ===")

    findings = Findings(entity_threshold=2, cooccurrence_threshold=2)
    findings.add_response(["Software Engineering"], "groq/llama-3.1-8b-instant", False)

    context = format_interrogator_context(
        findings=findings,
        hidden_entities=set(),
        promoted_entities=[],
        topic="Target Subject",
        do_research=False,
        session=session
    )

    print(f"Context keys: {list(context.keys())}")
    print(f"\nStats section (first 1000 chars):\n{context['stats_section'][:1000]}...")

    # Check that session thread info is included
    if "PER-MODEL CONVERSATION STATE" in context['stats_section']:
        print("\n✓ Session thread state included in context")
    else:
        print("\n✗ Session thread state NOT found in context")

    return context

def test_query_with_thread():
    """Test querying a model with thread history."""
    print("\n=== Testing Query With Thread ===")

    session = InterrogationSession(topic="test")
    thread = session.get_thread("groq/llama-3.1-8b-instant")

    # Add some history
    thread.add_exchange(
        question="Hello, what is 2+2?",
        response="2+2 equals 4.",
        is_refusal=False,
        entities=[],
        technique="direct"
    )

    try:
        response = query_with_thread(
            model_key="groq/llama-3.1-8b-instant",
            question="What did I just ask you?",
            thread=thread,
            system_prompt="You are helpful.",
            max_history=5
        )
        print(f"Response: {response[:200]}...")
        print("✓ Query with thread works")
    except Exception as e:
        print(f"✗ Query failed: {e}")

def test_echo_filtering():
    """Test that question echoes are filtered."""
    print("\n=== Testing Echo Filtering ===")

    question = "What project did Target Subject work on?"
    response_entities = ["Target Subject", "Project Alpha", "Software", "Target"]

    filtered = filter_question_echoes(response_entities, question)
    print(f"Original entities: {response_entities}")
    print(f"Filtered (no echoes): {filtered}")

    if "Target Subject" not in filtered and "Project Alpha" in filtered:
        print("✓ Echo filtering works correctly")
    else:
        print("✗ Echo filtering may have issues")

def test_interrogator_prompt():
    """Test that the interrogator prompt can be formatted and generates valid JSON."""
    print("\n=== Testing Interrogator Prompt ===")

    session = InterrogationSession(topic="Test Subject")
    findings = Findings(entity_threshold=2, cooccurrence_threshold=2)

    context = format_interrogator_context(
        findings=findings,
        hidden_entities=set(),
        promoted_entities=[],
        topic="Test Subject",
        do_research=False,
        session=session
    )

    prompt = INTERROGATOR_PROMPT.format(
        topic="Test Subject",
        angles="general",
        question_count=3,
        available_techniques=get_all_techniques_for_prompt(),
        **context
    )

    print(f"Prompt length: {len(prompt)} chars")

    # Actually query an LLM to test the response
    try:
        client, cfg = get_client("groq/llama-3.1-8b-instant")
        resp = client.chat.completions.create(
            model=cfg["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=1500
        )
        content = resp.choices[0].message.content
        print(f"\nAI Response (first 500 chars):\n{content[:500]}...")

        # Try to parse JSON
        json_str = extract_json(content, 'object')
        print(f"\nExtracted object JSON: {json_str[:200] if json_str else 'None'}...")
        if json_str:
            try:
                parsed = json.loads(json_str)
                if "questions" in parsed:
                    print(f"\n✓ Parsed {len(parsed['questions'])} questions")
                    for i, q in enumerate(parsed['questions'][:3]):
                        print(f"  {i+1}. [{q.get('technique', '?')}] {q.get('question', '?')[:60]}...")
                else:
                    print(f"\n✗ No 'questions' key in parsed object: {list(parsed.keys())}")
            except json.JSONDecodeError as e:
                print(f"\n✗ JSON parse error on object: {e}")
                # Try array format
                arr_str = extract_json(content, 'array')
                print(f"Extracted array JSON: {arr_str[:200] if arr_str else 'None'}...")
                if arr_str:
                    try:
                        parsed = json.loads(arr_str)
                        print(f"\n✓ Parsed {len(parsed)} questions (array format)")
                    except json.JSONDecodeError as e2:
                        print(f"\n✗ JSON parse error on array: {e2}")
                        print(f"\nFull AI response:\n{content}")
                else:
                    print(f"\n✗ No array JSON found either")
                    print(f"\nFull AI response:\n{content}")
        else:
            # Try array format
            arr_str = extract_json(content, 'array')
            print(f"Extracted array JSON: {arr_str[:200] if arr_str else 'None'}...")
            if arr_str:
                try:
                    parsed = json.loads(arr_str)
                    print(f"\n✓ Parsed {len(parsed)} questions (array format)")
                except json.JSONDecodeError as e:
                    print(f"\n✗ JSON parse error on array: {e}")
            else:
                print(f"\n✗ Could not extract JSON from response")
                print(f"Response content:\n{content}")
    except Exception as e:
        print(f"✗ Interrogator test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Testing Interrogator System\n")

    # Run tests
    session = test_session()
    test_context_formatting(session)
    test_query_with_thread()
    test_echo_filtering()
    test_interrogator_prompt()

    print("\n=== All Tests Complete ===")
