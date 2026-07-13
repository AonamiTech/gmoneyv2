from pydantic import Field, model_validator

from gmoney.contracts.common import ContractModel


class Point(ContractModel):
    x: float = Field(ge=0)
    y: float = Field(ge=0)


class Polygon(ContractModel):
    points: tuple[Point, ...]

    @model_validator(mode="after")
    def require_polygon(self) -> "Polygon":
        if len(self.points) < 3:
            raise ValueError("a polygon needs at least three points")
        return self


class TransformChain(ContractModel):
    page_number: int = Field(ge=1)
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    derived_width: int = Field(gt=0)
    derived_height: int = Field(gt=0)
    forward_matrix: tuple[tuple[float, float, float], ...]
    inverse_matrix: tuple[tuple[float, float, float], ...]
    operations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def require_three_by_three_matrices(self) -> "TransformChain":
        for matrix in (self.forward_matrix, self.inverse_matrix):
            if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
                raise ValueError("transform matrices must be 3x3")
        return self

