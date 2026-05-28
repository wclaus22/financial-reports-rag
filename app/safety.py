"""
Safety mechanisms for the generation process

Strategy pattern to implement an object that will implement different safety
checks on the retrieved hits and collections of hits from the vector DB,
to be used in the Generator class before sending the hits to the LLM for
answer generation.
"""

from typing import Protocol

from app.models import RetrievedHit


class SafetyMechanism(Protocol):
    """
    Protocol for safety mechanism to implement different safety checks on the retrieved hits and collections of hits from the vector DB
    """

    def add_hit(
        self, hit: RetrievedHit, filtered_hits: list[RetrievedHit]
    ) -> list[RetrievedHit]:
        """
        add a hit to the collection of hits to be sent to the LLM, with safety checks
        """
        ...

    def check_hits(self, filtered_hits: list[RetrievedHit]) -> list[RetrievedHit]:
        """
        check the collection of hits to be sent to the LLM, with safety checks
        """
        ...


class GenerationSafety:
    """
    safety class strategy to implement different safety checks on the
    retrieved hits and collections of hits from the vector DB
    """

    def __init__(self, mechanism: SafetyMechanism) -> None:
        self.mechanism = mechanism

    def add_hit(
        self, hit: RetrievedHit, filtered_hits: list[RetrievedHit]
    ) -> list[RetrievedHit]:
        """
        add a hit to the collection of hits to be sent to the LLM, with safety checks
        """
        return self.mechanism.add_hit(hit, filtered_hits)

    def check_hits(self, filtered_hits: list[RetrievedHit]) -> list[RetrievedHit]:
        """
        check the collection of hits to be sent to the LLM, with safety checks
        """
        return self.mechanism.check_hits(filtered_hits)


class NoSafetyMechanism(SafetyMechanism):
    """
    no safety mechanism, just add the hit and return the hits as is
    """

    def add_hit(
        self, hit: RetrievedHit, filtered_hits: list[RetrievedHit]
    ) -> list[RetrievedHit]:
        filtered_hits.append(hit)
        return filtered_hits

    def check_hits(self, filtered_hits: list[RetrievedHit]) -> list[RetrievedHit]:
        return filtered_hits
