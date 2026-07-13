from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class VersionedContract(ContractModel):
    id: UUID = Field(default_factory=uuid4)
    contract_version: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

