"""
Research Adapters Package.

Add new adapters here and they'll be auto-discovered.
"""

from .documentcloud import DocumentCloudAdapter
from .web_search import WebSearchAdapter

# Register all adapters
ALL_ADAPTERS = [
    DocumentCloudAdapter,
    WebSearchAdapter,
]
