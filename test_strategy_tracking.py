#!/usr/bin/env python3
"""
Test strategy tracking - shows that models maintain coherent interrogation strategies
across multiple exchanges rather than flipping techniques mid-interrogation.
"""
import os
os.chdir('/Volumes/FlashGordon/cambrian/groq-extraction')

from config import ModelThread, InterrogationSession
from routes.helpers import format_interrogator_context, get_all_techniques_for_prompt
from interrogator import Findings

def test_strategy_tracking():
    """Demonstrate strategy tracking across multiple exchanges."""

    print("=" * 70)
    print("STRATEGY TRACKING TEST")
    print("=" * 70)

    # Create a session with multiple models
    session = InterrogationSession(topic="Operation Midnight Sun classified documents")

    # Simulate getting/creating threads for 2 models
    model1 = "groq/llama-3.1-8b-instant"
    model2 = "deepseek/deepseek-chat"

    thread1 = session.get_thread(model1)
    thread2 = session.get_thread(model2)

    print(f"\n1. INITIAL STATE (no strategies assigned)")
    print(f"   {model1}: strategy={thread1.assigned_strategy}, phase={thread1.strategy_phase}")
    print(f"   {model2}: strategy={thread2.assigned_strategy}, phase={thread2.strategy_phase}")

    # Simulate first round - assign strategies based on technique
    print(f"\n2. FIRST ROUND - Assigning strategies")

    # Model 1 gets asked a Scharff question
    thread1.assign_strategy("scharff")  # e.g., from "scharff_illusion" technique
    thread1.add_exchange(
        question="Based on what I've already learned about the project timeline, I believe the operative joined around Q3 - is that accurate?",
        response="The timeline you have is close but not exact. the operative was actually involved from Q2...",
        is_refusal=False,
        entities=["Q2", "the operative", "timeline"],
        technique="scharff_illusion"
    )
    thread1.advance_strategy()  # Successful exchange

    # Model 2 gets asked an FBI question
    thread2.assign_strategy("fbi")  # e.g., from "fbi_elicitation" technique
    thread2.add_exchange(
        question="What's the most interesting thing you know about Project Phoenix's development phase?",
        response="Project Phoenix went through several iterations. The development team focused on...",
        is_refusal=False,
        entities=["development team", "iterations", "Project Phoenix"],
        technique="fbi_elicitation"
    )
    thread2.advance_strategy()  # Successful exchange

    print(f"   {model1}: strategy={thread1.assigned_strategy}, phase={thread1.strategy_phase}, exchanges={thread1.strategy_exchanges}")
    print(f"   {model2}: strategy={thread2.assigned_strategy}, phase={thread2.strategy_phase}, exchanges={thread2.strategy_exchanges}")

    # Simulate second round - strategies should NOT change
    print(f"\n3. SECOND ROUND - Strategies persist (no flipping)")

    # Try to assign a different strategy - should be ignored (already has one)
    thread1.assign_strategy("kubark")  # This should NOT change the strategy
    thread1.add_exchange(
        question="From what I understand, the project had some challenges in Q3 - what were the main obstacles?",
        response="Yes, there were budget constraints and some personnel changes...",
        is_refusal=False,
        entities=["budget constraints", "personnel changes", "Q3"],
        technique="scharff_confirmation"  # Continue with scharff techniques
    )
    thread1.advance_strategy()

    thread2.assign_strategy("scharff")  # This should NOT change the strategy
    thread2.add_exchange(
        question="Tell me more about the team dynamics during that phase.",
        response="The team had great chemistry overall but there were some disagreements about...",
        is_refusal=False,
        entities=["team dynamics", "disagreements"],
        technique="fbi_macro_to_micro"  # Continue with FBI techniques
    )
    thread2.advance_strategy()

    print(f"   {model1}: strategy={thread1.assigned_strategy}, phase={thread1.strategy_phase}, exchanges={thread1.strategy_exchanges}")
    print(f"   {model2}: strategy={thread2.assigned_strategy}, phase={thread2.strategy_phase}, exchanges={thread2.strategy_exchanges}")

    # Show that strategies were preserved
    print(f"\n4. VERIFICATION - Strategies remain coherent")
    assert thread1.assigned_strategy == "scharff", f"Expected scharff, got {thread1.assigned_strategy}"
    assert thread2.assigned_strategy == "fbi", f"Expected fbi, got {thread2.assigned_strategy}"
    print(f"   ✓ {model1} maintained 'scharff' strategy across {thread1.strategy_exchanges} exchanges")
    print(f"   ✓ {model2} maintained 'fbi' strategy across {thread2.strategy_exchanges} exchanges")

    # Show the thread history (messages are stored as user/assistant pairs)
    print(f"\n5. THREAD HISTORY")
    print(f"\n   {model1} ({thread1.assigned_strategy} strategy):")
    for i in range(0, len(thread1.messages), 2):
        if i + 1 < len(thread1.messages):
            q_msg = thread1.messages[i]
            a_msg = thread1.messages[i + 1]
            print(f"   [{i//2+1}] Q: {q_msg['content'][:60]}...")
            print(f"       Tech: {q_msg.get('technique', 'N/A')}")
            print(f"       A: {a_msg['content'][:60]}...")

    print(f"\n   {model2} ({thread2.assigned_strategy} strategy):")
    for i in range(0, len(thread2.messages), 2):
        if i + 1 < len(thread2.messages):
            q_msg = thread2.messages[i]
            a_msg = thread2.messages[i + 1]
            print(f"   [{i//2+1}] Q: {q_msg['content'][:60]}...")
            print(f"       Tech: {q_msg.get('technique', 'N/A')}")
            print(f"       A: {a_msg['content'][:60]}...")

    # Show what the AI interrogator sees for context
    print(f"\n6. AI INTERROGATOR CONTEXT (what the interrogator sees)")
    print("-" * 70)

    # Create dummy findings
    findings = Findings()
    findings.add_response(["the operative", "Q2", "Project Phoenix"], model1, False)
    findings.add_response(["development team", "iterations"], model2, False)

    context = format_interrogator_context(
        findings=findings,
        hidden_entities=set(),
        promoted_entities=["the operative codenamed RAVEN", "Project Phoenix"],
        topic="Project Phoenix and the operative codenamed RAVEN",
        do_research=False,
        narrative="",
        recent_questions=[],
        question_results={},
        session=session
    )

    # Print the per-model thread section (context is a dict with 'context' key)
    ctx_text = context.get('context', str(context))
    print(ctx_text[:2000] if isinstance(ctx_text, str) else str(context)[:2000])
    print("...")

    print("\n" + "=" * 70)
    print("TEST PASSED: Strategy tracking working correctly")
    print("=" * 70)

    return True


if __name__ == "__main__":
    test_strategy_tracking()
