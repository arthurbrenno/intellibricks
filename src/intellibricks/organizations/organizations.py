# Organizations are groups of agents.

from typing import Any, Sequence
import msgspec

from .agents import Agent


class Department(msgspec.Struct, frozen=True):
    """
    A department is a group of agents.
    When a problem comes to be solved, it will come
    to the department, the department will
    then delegate the problem to the specific
    agent(s) that can better solve this task.
    The metadata, task, instructions of
    each agent will help to identific
    the best(s) one(s) to solve the
    problem.
    """

    name: str
    description: str
    agents: Sequence[Agent]
    metadata: dict[str, Any] = msgspec.field(default_factory=dict)


class Organization(msgspec.Struct, frozen=True):
    """
    Essentially, an organization is a group of agents.
    When a problem comes to be solved, it will come
    to the organization, the organization will
    then delegate the problem to the specific
    agent(s) that can better solve this task.
    The metadata, task, instructions of
    each agent will help to identific
    the best(s) one(s) to solve the
    problem.


    Note(arthur):
    An organization is maybe a group of sectors? Than each sector
    is a group of agents. Each sector can have a leader, and the
    organization can have a leader. The organization leader
    can delegate tasks to the sector leaders, and the sector
    leaders can delegate tasks to the sector agents. I think
    that will be a really solid structure.

    Each department/element of the organization should be completely
    documentated and filled with metadata. This will help the subtasks
    to be delegated to the correct agents. Because there will be a lot
    of metadata to choose the correct agent to solve the problem.
    I think, in this case, there is no problem the metadata to be
    Mapping[str, Any] because it can be any metadata that the
    organization leader thinks that is important to choose the
    correct agent to solve the problem. Actually, this parameter should be
    called extra_metadata, because the most common metadatas should be instance defined attributes
    """

    name: str
    description: str
    CEO: Agent
    departments: Sequence[Department]
    metadata: dict[str, Any] = msgspec.field(default_factory=dict)


# the evolution should be just like in real life: Agents -> Sectors -> Organizations -> Government
