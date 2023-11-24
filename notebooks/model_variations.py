# see: https://stackoverflow.com/questions/33533148
from __future__ import annotations

from typing import Set, Tuple, Dict

from rethon import StandardGlobalReflectiveEquilibrium, StandardLocalReflectiveEquilibrium
from tau import DialecticalStructure, Position, DAGDialecticalStructure


class StandardGlobalReflectiveEquilibriumLinearG(StandardGlobalReflectiveEquilibrium):

    # overwrite model name initialization (otherwise ensemble data will be labelled wrongly!)
    def __init__(self, dialectical_structure: DialecticalStructure = None,
                 initial_commitments: Position = None,
                 model_name: str = "StandardGlobalReflectiveEquilibriumLinearG"):
        super().__init__(dialectical_structure, initial_commitments, model_name)

    def systematicity(self, theory: Position) -> float:
        if theory.size() == 0:
            return 0
        else:
            return 1 - ((theory.size() - 1) / self.dialectical_structure().closure(theory).size())

    def account(self, commitments: Position, theory: Position) -> float:
        return 1 - ((self.hamming_distance(commitments,
                                           self.dialectical_structure().closure(theory),
                                           self.model_parameter("account_penalties"))
                     / self.dialectical_structure().sentence_pool().size()))

    def faithfulness(self, commitments: Position, initial_commitments: Position) -> float:
        return 1 - ((self.hamming_distance(initial_commitments,
                                           commitments,
                                           self.model_parameter("faithfulness_penalties"))
                     / self.dialectical_structure().sentence_pool().size()))

class StandardLocalReflectiveEquilibriumLinearG(StandardLocalReflectiveEquilibrium):

    # overwrite model name initialization (otherwise ensemble data will be labelled wrongly!)
    def __init__(self, dialectical_structure: DialecticalStructure = None,
                 initial_commitments: Position = None,
                 model_name: str = "StandardLocalReflectiveEquilibriumLinearG"):
        super().__init__(dialectical_structure, initial_commitments, model_name)

    def systematicity(self, theory: Position) -> float:
        if theory.size() == 0:
            return 0
        else:
            return 1 - ((theory.size() - 1) / self.dialectical_structure().closure(theory).size())

    def account(self, commitments: Position, theory: Position) -> float:
        return 1 - ((self.hamming_distance(commitments,
                                           self.dialectical_structure().closure(theory),
                                           self.model_parameter("account_penalties"))
                     / self.dialectical_structure().sentence_pool().size()))

    def faithfulness(self, commitments: Position, initial_commitments: Position) -> float:
        return 1 - ((self.hamming_distance(initial_commitments,
                                           commitments,
                                           self.model_parameter("faithfulness_penalties"))
                     / self.dialectical_structure().sentence_pool().size()))


class StandardLocalReflectiveEquilibriumWithGO(StandardLocalReflectiveEquilibrium):

    def global_optima(self, initial_commitments: Position) -> Set[Tuple[Position, Position]]:
        dag_ds = DAGDialecticalStructure.from_arguments(self.dialectical_structure().get_arguments(),
                                                        self.dialectical_structure().sentence_pool().size())

        global_re = StandardGlobalReflectiveEquilibrium(dag_ds)
        global_re.set_model_parameters(self.model_parameters())

        return global_re.global_optima(initial_commitments)


class StandardLocalReflectiveEquilibriumLinearGWithGO(StandardLocalReflectiveEquilibriumLinearG):

    def global_optima(self, initial_commitments: Position) -> Set[Tuple[Position, Position]]:
        dag_ds = DAGDialecticalStructure.from_arguments(self.dialectical_structure().get_arguments(),
                                                        self.dialectical_structure().sentence_pool().size())

        global_re = StandardGlobalReflectiveEquilibriumLinearG(dag_ds)
        global_re.set_model_parameters(self.model_parameters())

        return global_re.global_optima(initial_commitments)
