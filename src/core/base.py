from pydantic import BaseModel, ConfigDict


class BaseSchema(BaseModel):
    """
    Base schema class with common configuration for all data models.
    """
    model_config = ConfigDict(
        from_attributes=True,
        use_enum_values=True,
        str_strip_whitespace=True,
        arbitrary_types_allowed=True
    )


__all__ = ["BaseSchema"]
