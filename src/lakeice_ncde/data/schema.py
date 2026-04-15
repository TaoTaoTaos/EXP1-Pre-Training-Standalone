from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class FeatureSchema:
    """Schema definition for the input channels and target."""

    time_channel: str
    feature_columns: list[str]
    input_channels: list[str]
    target_column: str
    target_transform: str

    def to_dict(self) -> dict:
        """Convert the schema to a serializable dictionary."""
        return asdict(self)
