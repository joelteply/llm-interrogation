"""
Research - external documents and queries.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class ResearchDocument(BaseModel):
    """A document from external research."""
    id: str
    source: str  # "documentcloud", "web", "archive", etc.
    title: str
    url: str
    content: str
    snippet: Optional[str] = None
    publish_date: Optional[datetime] = None
    fetched_at: datetime = Field(default_factory=datetime.now)

    # Quality signals
    is_useful: bool = True
    rejection_reason: Optional[str] = None

    @property
    def summary(self) -> str:
        """Short summary for display."""
        return self.snippet or self.content[:200]


class ResearchQuery(BaseModel):
    """A research query and its results."""
    query: str
    suggested_by: str = "user"  # "user", "llm", "entity_expansion"
    executed_at: datetime = Field(default_factory=datetime.now)

    # Results
    documents: list[ResearchDocument] = Field(default_factory=list)
    cached_count: int = 0
    fetched_count: int = 0

    @property
    def total_results(self) -> int:
        return len(self.documents)


class ResearchContext(BaseModel):
    """Accumulated research for a project."""
    queries: list[ResearchQuery] = Field(default_factory=list)
    documents: list[ResearchDocument] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=datetime.now)

    @property
    def useful_documents(self) -> list[ResearchDocument]:
        """Documents marked as useful."""
        return [d for d in self.documents if d.is_useful]

    def add_document(self, doc: ResearchDocument) -> bool:
        """Add document if not duplicate."""
        if any(d.url == doc.url for d in self.documents):
            return False
        self.documents.append(doc)
        self.updated_at = datetime.now()
        return True

    def get_context_for_query(self, terms: list[str], max_docs: int = 3) -> str:
        """Get relevant document snippets for query terms."""
        scored = []
        terms_lower = [t.lower() for t in terms]

        for doc in self.useful_documents:
            content_lower = doc.content.lower()
            score = sum(1 for t in terms_lower if t in content_lower)
            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)

        snippets = []
        for _, doc in scored[:max_docs]:
            snippets.append(f"[{doc.source}: {doc.title}]\n{doc.summary}")

        return "\n\n".join(snippets)
