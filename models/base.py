"""
Base entity classes.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class TimestampMixin(BaseModel):
    """Mixin for created/updated timestamps."""
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class BaseEntity(TimestampMixin):
    """
    Base for all persistent entities.

    Subclasses define their own id field with appropriate type.
    """
    model_config = ConfigDict(
        validate_assignment=True,
        extra="ignore",  # Ignore unknown fields during migration
        str_strip_whitespace=True,
    )

    def touch(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now()
