"""
Abstract class for any policy
"""

from abc import ABC, abstractmethod


class Policy(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def sample(self, s):
        """
        :return: sample of an action at state s.
        """
        pass
