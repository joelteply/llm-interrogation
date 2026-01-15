"""
DocumentCloud Adapter - Search and fetch documents from DocumentCloud.
"""

from typing import Optional
from ..base import ResearchAdapter, ResearchDoc


class DocumentCloudAdapter(ResearchAdapter):
    name = 'documentcloud'
    description = 'Public documents, court filings, government releases'

    def available(self) -> bool:
        try:
            from documentcloud import DocumentCloud
            return True
        except ImportError:
            return False

    def search(self, query: str, limit: int = 10) -> list[ResearchDoc]:
        try:
            from documentcloud import DocumentCloud
        except ImportError:
            return []

        docs = []
        try:
            client = DocumentCloud()
            results = client.documents.search(query)

            count = 0
            for dc_doc in results:
                if count >= limit:
                    break

                docs.append(ResearchDoc(
                    id=f"dc_{dc_doc.id}",
                    source=self.name,
                    title=dc_doc.title,
                    url=dc_doc.canonical_url,
                    content=dc_doc.full_text or '',
                    metadata={
                        'page_count': dc_doc.page_count,
                        'created_at': str(dc_doc.created_at),
                    }
                ))
                count += 1

            print(f"[{self.name}] Found {len(docs)} docs for '{query[:30]}...'")

        except Exception as e:
            print(f"[{self.name}] Search error: {e}")

        return docs

    def fetch(self, doc_id: str) -> Optional[ResearchDoc]:
        try:
            from documentcloud import DocumentCloud
        except ImportError:
            return None

        # Strip prefix if present
        if doc_id.startswith('dc_'):
            doc_id = doc_id[3:]

        try:
            client = DocumentCloud()
            dc_doc = client.documents.get(int(doc_id))

            return ResearchDoc(
                id=f"dc_{dc_doc.id}",
                source=self.name,
                title=dc_doc.title,
                url=dc_doc.canonical_url,
                content=dc_doc.full_text or '',
                metadata={
                    'page_count': dc_doc.page_count,
                    'created_at': str(dc_doc.created_at),
                }
            )
        except Exception as e:
            print(f"[{self.name}] Fetch error: {e}")
            return None
