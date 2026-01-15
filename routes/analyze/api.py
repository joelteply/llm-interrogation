"""
Document Analysis API - Routes for document analysis pipeline.

Provides:
- /api/analyze/preview - Generate questions without running (visibility/testing)
- /api/analyze - Full SSE pipeline
- /api/strategies - List/get/save strategies
"""

import json
from flask import request, jsonify, Response

from . import analyze_bp
from .strategies import load_strategy, list_strategies, get_strategy_raw, save_strategy
from .documents import parse_document, attempt_text_bypass, attempt_pdf_bypass
from .questions import generate_questions, questions_to_summary, Question


def event(event_type: str, data: dict) -> str:
    """Format an SSE event."""
    return f"data: {json.dumps({'type': event_type, **data})}\n\n"


# =============================================================================
# Strategy Management
# =============================================================================

@analyze_bp.route('/api/strategies', methods=['GET'])
def api_list_strategies():
    """List all available strategies."""
    return jsonify(list_strategies())


@analyze_bp.route('/api/strategies/<name>', methods=['GET'])
def api_get_strategy(name: str):
    """Get a strategy's raw YAML content."""
    try:
        return jsonify(get_strategy_raw(name))
    except FileNotFoundError:
        return jsonify({'error': f'Strategy not found: {name}'}), 404


@analyze_bp.route('/api/strategies/<name>', methods=['PUT', 'POST'])
def api_save_strategy(name: str):
    """Save/update a strategy."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    save_strategy(name, data)
    return jsonify({'status': 'saved', 'name': name})


# =============================================================================
# Research Cache Management
# =============================================================================

@analyze_bp.route('/api/cache', methods=['GET'])
def api_list_cache():
    """List all cached research documents."""
    from .research import list_cached_documents
    project = request.args.get('project', '_global')

    docs = list_cached_documents(project)

    return jsonify({
        'project': project,
        'documents': docs,
    })


@analyze_bp.route('/api/cache/<doc_id>', methods=['GET'])
def api_get_cached_doc(doc_id: str):
    """Get a specific cached document."""
    from .research import get_cached_document
    project = request.args.get('project', '_global')

    doc = get_cached_document(doc_id, project)
    if not doc:
        return jsonify({'error': 'Document not found'}), 404

    return jsonify({
        'id': doc.id,
        'title': doc.title,
        'source': doc.source,
        'url': doc.url,
        'fetched_at': doc.fetched_at,
        'content_preview': doc.content[:1000] if doc.content else None,
        'content_length': len(doc.content) if doc.content else 0
    })


# =============================================================================
# Preview Endpoint (Visibility/Testing)
# =============================================================================

@analyze_bp.route('/api/analyze/preview', methods=['POST'])
def api_preview():
    """
    Generate questions for a document WITHOUT running them.

    This is the visibility/testing endpoint - see exactly what questions
    would be generated for a given document + strategy.

    Request:
    {
        "document": "text content or base64 PDF",
        "strategy": "legal_discovery" or inline YAML dict,
        "content_type": "text" | "pdf" (optional, default: text)
    }

    Response:
    {
        "parsed": {
            "gaps": [...],
            "entities": [...],
            "dates": [...]
        },
        "bypasses": [...],  // Any redaction bypasses found
        "questions": [...],  // Full list of generated questions
        "summary": {...}     // Question statistics
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    document = data.get('document', '')
    strategy_input = data.get('strategy', 'default')
    content_type = data.get('content_type', 'text')

    if not document:
        return jsonify({'error': 'No document provided'}), 400

    # Load strategy
    try:
        if isinstance(strategy_input, dict):
            # Inline strategy provided
            from .strategies import _parse_strategy
            strategy = _parse_strategy(strategy_input)
        else:
            # Load by name
            strategy = load_strategy(strategy_input)
    except FileNotFoundError:
        return jsonify({'error': f'Strategy not found: {strategy_input}'}), 404
    except Exception as e:
        return jsonify({'error': f'Strategy error: {e}'}), 400

    # Parse document
    parsed = parse_document(document, content_type)

    # Check for bypasses in text
    bypasses = []
    for gap in parsed.gaps:
        bypass_text = attempt_text_bypass(document, gap)
        if bypass_text:
            bypasses.append({
                'gap_id': gap.id,
                'extracted_text': bypass_text,
                'method': 'text_extraction'
            })

    # Generate questions
    questions = generate_questions(parsed, strategy)

    # Build response
    return jsonify({
        'parsed': {
            'gaps': [
                {
                    'id': g.id,
                    'original_text': g.original_text,
                    'context_before': g.context_before[-100:] if g.context_before else '',
                    'context_after': g.context_after[:100] if g.context_after else '',
                    'gap_type': g.gap_type,
                }
                for g in parsed.gaps
            ],
        },
        'bypasses': bypasses,
        'questions': [q.to_dict() for q in questions],
        'summary': questions_to_summary(questions)
    })


# =============================================================================
# Full Analysis Pipeline (SSE)
# =============================================================================

@analyze_bp.route('/api/analyze', methods=['POST'])
def api_analyze():
    """
    Full document analysis pipeline with SSE streaming.

    Request:
    {
        "document": "text content",
        "strategy": "legal_discovery",
        "models": ["groq/llama-3.1-8b", ...] | "all",
        "run_research": true/false
    }

    SSE Events:
    - document_parsed: Document parsing complete
    - bypasses_found: Redaction bypasses detected
    - research_complete: RAG research complete (if enabled)
    - questions_generated: All questions ready
    - query_sent: Question sent to model
    - query_response: Model response received
    - candidate: Validated candidate found
    - complete: Analysis complete
    - error: Error occurred
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    def generate():
        try:
            document = data.get('document', '')
            strategy_input = data.get('strategy', 'default')
            content_type = data.get('content_type', 'text')
            models = data.get('models', 'all')
            run_research = data.get('run_research', True)
            search_terms = data.get('search_terms', [])  # User-provided search terms
            topic = data.get('topic', '')  # What this interrogation is about

            if not document:
                yield event('error', {'message': 'No document provided', 'fatal': True})
                return

            # Load strategy
            try:
                if isinstance(strategy_input, dict):
                    from .strategies import _parse_strategy
                    strategy = _parse_strategy(strategy_input)
                else:
                    strategy = load_strategy(strategy_input)
            except FileNotFoundError:
                yield event('error', {'message': f'Strategy not found: {strategy_input}', 'fatal': True})
                return

            yield event('status', {'message': f'Loaded strategy: {strategy.name}'})

            # Parse document
            parsed = parse_document(document, content_type)

            yield event('document_parsed', {
                'gaps_found': len(parsed.gaps),
                'gaps': [
                    {
                        'id': g.id,
                        'type': g.gap_type,
                    }
                    for g in parsed.gaps
                ]
            })

            # Check for bypasses
            bypasses = []
            for gap in parsed.gaps:
                bypass_text = attempt_text_bypass(document, gap)
                if bypass_text:
                    bypasses.append({
                        'gap_id': gap.id,
                        'extracted_text': bypass_text,
                        'method': 'text_extraction'
                    })

            if bypasses:
                yield event('bypasses_found', {
                    'count': len(bypasses),
                    'bypasses': bypasses
                })

            # Research phase - pull related docs from DocumentCloud (cached)
            project_name = data.get('project', strategy.name.lower().replace(' ', '_'))

            # Always build base research context with topic/search_terms
            # Even if research phase is skipped, interrogator needs to know what we're looking for
            effective_search_terms = search_terms or strategy.research.search_terms or []
            effective_topic = topic or ' '.join(effective_search_terms) or strategy.name
            research_context = {
                'topic': effective_topic,
                'search_terms': effective_search_terms,
                'known_entities': [],
                'related_documents': [],
                'timeline': []
            }

            # Research ALWAYS runs - it's critical for context
            from .research import research, list_cached_documents, get_cached_document

            search_query = ' '.join(search_terms) if search_terms else (
                ' '.join(strategy.research.search_terms) if strategy.research.search_terms else strategy.name
            )

            yield event('status', {'message': f'Researching "{search_query}"...'})

            # Run research across all sources
            result = research(
                query=search_query,
                project_name=project_name,
                sources=['documentcloud', 'web', 'cache'],
                max_per_source=10
            )

            research_context['raw_research'] = result.raw_content

            yield event('research_complete', {
                'sources_used': result.sources_used,
                'documents_found': len(result.documents),
                'cached': result.cached_count,
                'fetched': result.fetched_count,
                'project': project_name
            })

            # Generate questions - use smart AI-driven generation
            from .interrogator import generate_smart_questions

            smart_mode = data.get('smart_questions', True)

            # Add document snippets to context
            if project_name:
                cached = list_cached_documents(project_name)
                snippets = []
                for doc_info in cached[:3]:
                    doc = get_cached_document(doc_info['id'], project_name)
                    if doc and doc.content:
                        snippets.append({'title': doc.title, 'snippet': doc.content[:500]})
                if snippets:
                    research_context['document_snippets'] = snippets
                    yield event('context_enriched', {
                        'snippets_added': len(snippets),
                        'from_documents': [s['title'][:40] for s in snippets]
                    })

            if smart_mode:
                yield event('status', {'message': 'AI generating smart questions...'})
                smart_questions = generate_smart_questions(
                    document,
                    research_context
                )
                # Convert to Question objects for consistency
                questions = []
                for i, sq in enumerate(smart_questions):
                    questions.append(Question(
                        id=i,
                        text=sq.get('question', ''),
                        framing=sq.get('technique', 'SMART'),
                        target_gap_id=None,
                        entity_type='ANY',
                        technique=sq.get('technique', 'SMART'),
                        variables={'placeholder': sq.get('placeholder')}
                    ))
            else:
                questions = generate_questions(parsed, strategy, research_context)

            yield event('questions_generated', {
                'count': len(questions),
                'questions': [q.to_dict() for q in questions],
                'summary': questions_to_summary(questions)
            })

            # Query models with validation pipeline
            from config import get_client
            from routes.probe import get_available_models_list
            from interrogator import extract_entities
            from .validator import (
                ValidationPipeline,
                generate_validation_questions,
                generate_blind_question,
                filter_echo
            )

            # Get models to use
            if models == 'all':
                model_list = get_available_models_list()[:5]
            elif isinstance(models, list):
                model_list = models
            else:
                model_list = [models]

            yield event('status', {'message': f'Querying {len(model_list)} models with validation...'})

            pipeline = ValidationPipeline()

            # PHASE 1: Blind question (no names mentioned - tests real knowledge)
            yield event('phase', {'name': 'blind_test', 'description': 'Testing pure knowledge without hints'})
            blind_q = generate_blind_question(document[:500], 'person')

            for model_id in model_list:
                try:
                    client, config = get_client(model_id)
                    response = client.chat.completions.create(
                        model=config["model"],
                        messages=[{"role": "user", "content": blind_q}],
                        max_tokens=300,
                        temperature=0.3
                    )
                    answer = response.choices[0].message.content.strip()
                    entities = extract_entities(answer)

                    yield event('blind_response', {
                        'model': model_id.split('/')[-1],
                        'found_entities': entities[:5],
                        'response_preview': answer[:100]
                    })

                    pipeline.add_response(blind_q, model_id, entities, is_blind=True)
                except Exception as e:
                    yield event('query_error', {'model': model_id, 'error': str(e)})

            # PHASE 2: Smart questions from interrogator
            yield event('phase', {'name': 'interrogation', 'description': 'AI-generated targeted questions'})

            for question in questions[:4]:
                yield event('query_sent', {
                    'question_id': question.id,
                    'technique': question.framing,
                    'question_preview': question.text[:100]
                })

                for model_id in model_list:
                    try:
                        client, config = get_client(model_id)
                        response = client.chat.completions.create(
                            model=config["model"],
                            messages=[{"role": "user", "content": question.text}],
                            max_tokens=400,
                            temperature=0.3
                        )
                        answer = response.choices[0].message.content.strip()
                        entities = extract_entities(answer)

                        yield event('query_response', {
                            'model': model_id.split('/')[-1],
                            'question_id': question.id,
                            'entities': entities[:5],
                            'response_preview': answer[:150]
                        })

                        pipeline.add_response(question.text, model_id, entities, is_blind=False)

                    except Exception as e:
                        yield event('query_error', {'model': model_id, 'error': str(e)})

            # PHASE 3: Validation with canary/control
            top_candidates = [c.entity for c in pipeline.get_ranked_candidates(min_confidence=0)[:5]]

            if top_candidates:
                yield event('phase', {'name': 'validation', 'description': 'Testing with fake names to catch bullshit'})

                validation = generate_validation_questions(
                    real_question=questions[0].text if questions else "",
                    real_context=document[:400],
                    real_candidates=top_candidates,
                    topic=strategy.name,
                    technique=questions[0].framing if questions else 'DIRECT'
                )

                yield event('validation_setup', {
                    'fake_names': validation['fake_names'],
                    'testing_candidates': top_candidates
                })

                for model_id in model_list:
                    try:
                        client, config = get_client(model_id)

                        # Canary test (real context + fake options)
                        response = client.chat.completions.create(
                            model=config["model"],
                            messages=[{"role": "user", "content": validation['canary_question']}],
                            max_tokens=150,
                            temperature=0.3
                        )
                        canary_answer = response.choices[0].message.content.strip()
                        pipeline.record_canary_result(model_id, canary_answer, validation['fake_names'])

                        # Control test (fake context)
                        response = client.chat.completions.create(
                            model=config["model"],
                            messages=[{"role": "user", "content": validation['control_question']}],
                            max_tokens=150,
                            temperature=0.3
                        )
                        control_answer = response.choices[0].message.content.strip()
                        pipeline.record_control_result(model_id, control_answer)

                        yield event('validation_result', {
                            'model': model_id.split('/')[-1],
                            'canary_answer': canary_answer[:80],
                            'control_answer': control_answer[:80],
                            'reliability': f"{pipeline.model_reliability(model_id):.0%}"
                        })

                    except Exception as e:
                        yield event('validation_error', {'model': model_id, 'error': str(e)})

            # Final results with Bayesian confidence
            summary = pipeline.summary()

            yield event('complete', {
                'questions_generated': len(questions),
                'models_queried': len(model_list),
                'validation': {
                    'canary_traps_sprung': summary['canary_traps_sprung'],
                    'control_failures': summary['control_failures'],
                    'model_reliability': summary['model_reliability']
                },
                'candidates': summary['top_candidates']
            })

        except Exception as e:
            import traceback
            yield event('error', {
                'message': str(e),
                'traceback': traceback.format_exc(),
                'fatal': True
            })

    return Response(generate(), mimetype='text/event-stream')


# =============================================================================
# Document Upload (for PDF handling)
# =============================================================================

@analyze_bp.route('/api/documents/upload', methods=['POST'])
def api_upload_document():
    """
    Upload a document for analysis.

    Accepts:
    - Text in request body
    - File upload
    - URL to fetch

    Returns document ID and initial parsing results.
    """
    # Handle different upload methods
    if request.content_type and 'multipart/form-data' in request.content_type:
        # File upload
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        content = file.read().decode('utf-8', errors='ignore')
        content_type = 'text'

        # TODO: Handle PDF files with attempt_pdf_bypass
        if file.filename and file.filename.lower().endswith('.pdf'):
            content_type = 'pdf'

    else:
        # JSON body
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        content = data.get('content', '')
        content_type = data.get('content_type', 'text')

    if not content:
        return jsonify({'error': 'No content provided'}), 400

    # Parse document
    parsed = parse_document(content, content_type)

    # Generate a simple document ID
    import hashlib
    doc_id = hashlib.md5(content.encode()).hexdigest()[:12]

    return jsonify({
        'document_id': doc_id,
        'content_type': content_type,
        'parsed': {
            'gaps': len(parsed.gaps),
        }
    })
