"""
DocumentCloud Adapter - Search court docs, government releases, etc.

Requires:
  DOCUMENTCLOUD_USERNAME
  DOCUMENTCLOUD_PASSWORD
"""

import os
from typing import Optional
from ..base import ResearchAdapter, ResearchDoc, AdapterConfig


class DocumentCloudAdapter(ResearchAdapter):
    name = 'documentcloud'
    description = 'Court filings, government docs, public records'

    def _default_config(self) -> AdapterConfig:
        return AdapterConfig(
            timeout=60,
            max_content_size=100000,
            env_vars={'DOCUMENTCLOUD_USERNAME': '', 'DOCUMENTCLOUD_PASSWORD': ''}
        )

    def available(self) -> bool:
        """Check if documentcloud package installed and credentials set."""
        try:
            from documentcloud import DocumentCloud
        except ImportError:
            print(f"[{self.name}] documentcloud package not installed")
            return False

        username = os.environ.get('DOCUMENTCLOUD_USERNAME')
        password = os.environ.get('DOCUMENTCLOUD_PASSWORD')

        if not username or not password:
            print(f"[{self.name}] Missing DOCUMENTCLOUD_USERNAME or DOCUMENTCLOUD_PASSWORD")
            return False

        return True

    def _get_client(self):
        """Get authenticated DocumentCloud client."""
        from documentcloud import DocumentCloud

        username = os.environ.get('DOCUMENTCLOUD_USERNAME')
        password = os.environ.get('DOCUMENTCLOUD_PASSWORD')

        if username and password:
            return DocumentCloud(username=username, password=password)
        return DocumentCloud()

    def search(self, query: str, limit: int = 10) -> list[ResearchDoc]:
        """Search DocumentCloud for documents."""
        if not self.available():
            return []

        try:
            client = self._get_client()
            print(f"[{self.name}] Searching: {query[:50]}...")

            results = client.documents.search(query)

            docs = []
            for i, dc_doc in enumerate(results):
                if i >= limit:
                    break

                # Get full text
                try:
                    full_text = dc_doc.full_text or ''
                except Exception:
                    full_text = ''

                if not full_text:
                    print(f"[{self.name}] No text for: {dc_doc.title[:40]}")
                    continue

                # Truncate if needed
                if len(full_text) > self.config.max_content_size:
                    full_text = full_text[:self.config.max_content_size] + '\n\n[TRUNCATED]'

                docs.append(ResearchDoc(
                    id=f"dc_{dc_doc.id}",
                    source=self.name,
                    title=dc_doc.title,
                    url=dc_doc.canonical_url,
                    content=full_text,
                    metadata={
                        'page_count': dc_doc.page_count,
                        'created_at': str(dc_doc.created_at) if dc_doc.created_at else '',
                        'contributor': dc_doc.contributor if hasattr(dc_doc, 'contributor') else '',
                    }
                ))
                print(f"[{self.name}] Got: {dc_doc.title[:50]} ({len(full_text)} chars)")

            print(f"[{self.name}] Returning {len(docs)} documents")
            return docs

        except Exception as e:
            print(f"[{self.name}] Search error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def fetch(self, doc_id: str) -> Optional[ResearchDoc]:
        """Fetch specific document by ID."""
        if not self.available():
            return None

        # Strip prefix
        if doc_id.startswith('dc_'):
            doc_id = doc_id[3:]

        try:
            client = self._get_client()
            dc_doc = client.documents.get(doc_id)

            full_text = dc_doc.full_text or ''

            return ResearchDoc(
                id=f"dc_{dc_doc.id}",
                source=self.name,
                title=dc_doc.title,
                url=dc_doc.canonical_url,
                content=full_text,
                metadata={
                    'page_count': dc_doc.page_count,
                    'created_at': str(dc_doc.created_at) if dc_doc.created_at else '',
                }
            )

        except Exception as e:
            print(f"[{self.name}] Fetch error: {e}")
            return None
