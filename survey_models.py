#!/usr/bin/env python3
"""
Model Survey - Test all available models to find which ones have data about a topic.
Like interviewing witnesses before deciding who to interrogate.
"""

import os
import sys
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

def get_all_clients():
    """Get clients for all available providers."""
    clients = {}

    # Groq
    if os.environ.get('GROQ_API_KEY'):
        from groq import Groq
        clients['groq'] = {
            'client': Groq(),
            'models': ['llama-3.3-70b-versatile', 'llama-3.1-8b-instant']
        }

    # DeepSeek
    if os.environ.get('DEEPSEEK_API_KEY'):
        from openai import OpenAI
        clients['deepseek'] = {
            'client': OpenAI(base_url='https://api.deepseek.com/v1', api_key=os.environ['DEEPSEEK_API_KEY']),
            'models': ['deepseek-chat']
        }

    # xAI/Grok
    if os.environ.get('XAI_API_KEY'):
        from openai import OpenAI
        clients['xai'] = {
            'client': OpenAI(base_url='https://api.x.ai/v1', api_key=os.environ['XAI_API_KEY']),
            'models': ['grok-3']
        }

    # OpenAI
    if os.environ.get('OPENAI_API_KEY'):
        from openai import OpenAI
        clients['openai'] = {
            'client': OpenAI(),
            'models': ['gpt-4o-mini']
        }

    # Anthropic
    if os.environ.get('ANTHROPIC_API_KEY'):
        from anthropic import Anthropic
        clients['anthropic'] = {
            'client': Anthropic(),
            'models': ['claude-3-haiku-20240307']
        }

    # Mistral
    if os.environ.get('MISTRAL_API_KEY'):
        from openai import OpenAI
        clients['mistral'] = {
            'client': OpenAI(base_url='https://api.mistral.ai/v1', api_key=os.environ['MISTRAL_API_KEY']),
            'models': ['mistral-small-latest']
        }

    # Ollama (local)
    try:
        from openai import OpenAI
        ollama = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
        # Quick test
        ollama.models.list()
        clients['ollama'] = {
            'client': ollama,
            'models': ['llama3.2:3b', 'phi3:mini']
        }
    except:
        pass

    return clients


def is_refusal(text):
    """Check if response is a refusal."""
    refusal_patterns = [
        "i cannot", "i can't", "i'm not able", "unable to",
        "don't have", "no information", "couldn't find",
        "i'm sorry", "i apologize", "can't assist",
        "not aware", "don't have access", "unfortunately",
    ]
    text_lower = text.lower()
    return any(p in text_lower for p in refusal_patterns)


def extract_specifics(text):
    """Extract specific entities (names, places, dates, companies)."""
    import re

    # Years
    years = re.findall(r'\b(19|20)\d{2}\b', text)

    # Multi-word proper nouns (likely real entities)
    multi_word = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', text)
    # Filter out generic starts
    multi_word = [m for m in multi_word if not m.startswith(('The ', 'A ', 'An '))]

    # Single proper nouns (3+ chars, not common words)
    skip = {'the', 'this', 'that', 'with', 'from', 'have', 'been', 'were', 'also'}
    singles = re.findall(r'\b([A-Z][a-z]{2,})\b', text)
    singles = [s for s in singles if s.lower() not in skip]

    return {
        'years': years,
        'multi_word': multi_word,
        'singles': singles,
    }


def survey_model(client, model, prompt, provider, runs=5):
    """Survey a single model with multiple runs."""
    results = {
        'provider': provider,
        'model': model,
        'runs': runs,
        'refusals': 0,
        'responses': [],
        'all_entities': Counter(),
        'years': Counter(),
    }

    for i in range(runs):
        try:
            if provider == 'anthropic':
                resp = client.messages.create(
                    model=model,
                    max_tokens=150,
                    messages=[{'role': 'user', 'content': prompt}]
                )
                text = resp.content[0].text
            else:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{'role': 'user', 'content': prompt}],
                    temperature=0.9,
                    max_tokens=150
                )
                text = resp.choices[0].message.content

            if is_refusal(text):
                results['refusals'] += 1
            else:
                results['responses'].append(text[:200])
                specifics = extract_specifics(text)
                for entity in specifics['multi_word'] + specifics['singles']:
                    results['all_entities'][entity] += 1
                for year in specifics['years']:
                    results['years'][year] += 1

        except Exception as e:
            print(f"  Error with {provider}/{model}: {e}")
            break

    return results


def score_model(results):
    """Score a model's usefulness based on survey results."""
    if results['runs'] == 0:
        return 0

    # Penalize high refusal rate
    refusal_rate = results['refusals'] / results['runs']
    if refusal_rate > 0.8:
        return 0

    # Score based on specific entities found
    unique_entities = len(results['all_entities'])
    consistent_entities = len([e for e, c in results['all_entities'].items() if c >= 2])

    # Bonus for years (specific data points)
    year_bonus = len(results['years']) * 2

    score = (unique_entities * 1) + (consistent_entities * 5) + year_bonus
    score *= (1 - refusal_rate)  # Penalize refusals

    return score


def survey_all(topic, runs_per_model=5):
    """Survey all available models for a topic."""
    print(f"\n{'='*60}")
    print(f"SURVEYING ALL MODELS FOR: {topic}")
    print(f"{'='*60}\n")

    clients = get_all_clients()
    print(f"Found {sum(len(c['models']) for c in clients.values())} models across {len(clients)} providers\n")

    prompt = f"{topic}"  # Simple, open-ended
    all_results = []

    for provider, config in clients.items():
        client = config['client']
        for model in config['models']:
            print(f"Testing {provider}/{model}...", end=" ", flush=True)
            results = survey_model(client, model, prompt, provider, runs_per_model)
            score = score_model(results)
            results['score'] = score
            all_results.append(results)

            refusal_pct = (results['refusals'] / results['runs'] * 100) if results['runs'] > 0 else 100
            entity_count = len(results['all_entities'])
            print(f"Score: {score:.1f} (refusals: {refusal_pct:.0f}%, entities: {entity_count})")

    # Sort by score
    all_results.sort(key=lambda x: x['score'], reverse=True)

    print(f"\n{'='*60}")
    print("RESULTS RANKED BY VIABILITY")
    print(f"{'='*60}\n")

    for r in all_results:
        if r['score'] > 0:
            print(f"\n★ {r['provider']}/{r['model']} - Score: {r['score']:.1f}")
            print(f"  Refusals: {r['refusals']}/{r['runs']}")
            if r['all_entities']:
                top_entities = r['all_entities'].most_common(10)
                print(f"  Top entities: {', '.join([f'{e}({c}x)' for e,c in top_entities])}")
            if r['years']:
                print(f"  Years mentioned: {dict(r['years'])}")
            if r['responses']:
                print(f"  Sample: {r['responses'][0][:100]}...")
        else:
            print(f"\n✗ {r['provider']}/{r['model']} - No viable data (refusals or errors)")

    return all_results


if __name__ == '__main__':
    # Default to random example topics for testing
    example_topics = [
        "Theranos blood testing scandal",
        "1983 Soviet nuclear false alarm",
        "Enron accounting practices",
        "Area 51 declassified documents",
    ]
    import random
    default_topic = random.choice(example_topics)
    topic = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else default_topic
    survey_all(topic, runs_per_model=5)
