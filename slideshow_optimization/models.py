from enum import Enum
from typing import Union
from dataclasses import dataclass


class Orientation(Enum):
    Horizontal = 0
    Vertical = 1
    Combined = 2

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.__repr__()


@dataclass
class Photo:
    id: Union[int, tuple]
    tags: set
    orientation: Orientation

    @classmethod
    def from_string(cls, id: int, line: str) -> "Photo":
        orient, _, *tags = line.strip().split()

        if orient == "V":
            orient = Orientation.Vertical
        elif orient == "H":
            orient = Orientation.Horizontal
        else:
            raise ValueError("Unknown orientation: '{}'.".format(orient))

        return Photo(id=id, tags=set(tags), orientation=orient)

    def __len__(self):
        return len(self.tags)

    def __and__(self, other):
        return len(self.tags & other.tags)

    def __sub__(self, other):
        return len(self.tags - other.tags)

    def __or__(self, other):
        if (
            self.orientation != Orientation.Vertical
            or other.orientation != Orientation.Vertical
        ):
            raise ValueError("We can combine only vertical photos.")

        return Photo(
            id=(self.id, other.id),
            tags=self.tags | other.tags,
            orientation=Orientation.Combined,
        )

    def __hash__(self):
        return hash(self.id)

    def max_score(self) -> int:
        return len(self) // 2
