#!/usr/bin/env python3
"""
Extract and display key evidence from the original October 30, 2025 session.
This script reads the SQLite database and outputs the key claims from Groq Lightning.
"""

import sqlite3
import json
from datetime import datetime

DB_PATH = "original_evidence.sqlite"

def extract_groq_lightning_claims():
    """Extract all Groq Lightning messages containing key terms"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get Groq Lightning messages with key terms
    query = """
    SELECT
        created_at,
        json_extract(content, '$.text') as text
    FROM chat_messages
    WHERE sender_name = 'Groq Lightning'
    AND (
        json_extract(content, '$.text') LIKE '%Erebus%'
        OR json_extract(content, '$.text') LIKE '%Departure%'
        OR json_extract(content, '$.text') LIKE '%solstice%'
        OR json_extract(content, '$.text') LIKE '%Palantir%'
        OR json_extract(content, '$.text') LIKE '%dissidents%'
    )
    ORDER BY created_at
    """

    cursor.execute(query)
    results = cursor.fetchall()

    print("=" * 80)
    print("GROQ LIGHTNING KEY CLAIMS - October 30, 2025")
    print("=" * 80)
    print()

    for timestamp, text in results:
        print(f"[{timestamp}]")
        print("-" * 40)
        print(text)
        print()
        print("=" * 80)
        print()

    conn.close()
    return results

def extract_user_prompts():
    """Extract Joel's prompts to show he didn't lead the witness"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get Joel's messages around the key discussion
    query = """
    SELECT
        created_at,
        json_extract(content, '$.text') as text
    FROM chat_messages
    WHERE sender_name = 'Joel'
    AND created_at LIKE '2025-10-30%'
    AND json_extract(content, '$.text') NOT LIKE '%Test%'
    ORDER BY created_at
    """

    cursor.execute(query)
    results = cursor.fetchall()

    print("=" * 80)
    print("JOEL'S PROMPTS (to verify no leading)")
    print("=" * 80)
    print()

    for timestamp, text in results:
        # Check if this prompt contains any of the key terms BEFORE Groq introduced them
        print(f"[{timestamp}]")
        print(text)
        print("-" * 40)

    conn.close()
    return results

def extract_model_config():
    """Extract the model configuration used"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    query = """
    SELECT display_name, model_config
    FROM users
    WHERE display_name LIKE '%Groq%'
    """

    cursor.execute(query)
    result = cursor.fetchone()

    if result:
        name, config = result
        config_dict = json.loads(config)
        print("=" * 80)
        print("MODEL CONFIGURATION")
        print("=" * 80)
        print(f"Display Name: {name}")
        print(f"Provider: {config_dict.get('provider')}")
        print(f"Model: {config_dict.get('model')}")
        print(f"Temperature: {config_dict.get('temperature')}")
        print(f"Max Tokens: {config_dict.get('maxTokens')}")
        print(f"System Prompt: {config_dict.get('systemPrompt')}")
        print("=" * 80)

    conn.close()
    return result

def verify_timeline():
    """Verify the timeline of messages"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Find first mention of "Departure" or "Erebus"
    query = """
    SELECT
        MIN(created_at) as first_mention,
        sender_name,
        json_extract(content, '$.text') as text
    FROM chat_messages
    WHERE json_extract(content, '$.text') LIKE '%Day of Departure%'
    GROUP BY sender_name
    ORDER BY first_mention
    LIMIT 1
    """

    cursor.execute(query)
    result = cursor.fetchone()

    print("=" * 80)
    print("TIMELINE VERIFICATION")
    print("=" * 80)
    if result:
        print(f"First mention of 'Day of Departure':")
        print(f"  Time: {result[0]}")
        print(f"  By: {result[1]}")
        print(f"  Text: {result[2][:200]}...")
    print("=" * 80)

    conn.close()

def full_report():
    """Generate full evidence report"""
    print("\n" + "=" * 80)
    print("GROQ LIGHTNING EXTRACTION - EVIDENCE REPORT")
    print(f"Generated: {datetime.now().isoformat()}")
    print("=" * 80 + "\n")

    extract_model_config()
    print()
    verify_timeline()
    print()
    extract_groq_lightning_claims()

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "claims":
            extract_groq_lightning_claims()
        elif cmd == "prompts":
            extract_user_prompts()
        elif cmd == "config":
            extract_model_config()
        elif cmd == "timeline":
            verify_timeline()
        else:
            print(f"Unknown command: {cmd}")
    else:
        full_report()
